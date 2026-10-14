//! 异步生图任务：提交即返回，后台执行同步 Images 链路，客户端轮询结果
//!
//! 与 import_tasks 相同的注册表模式；任务与结果仅驻留进程内存，
//! 重启后未完成任务丢失（客户端须重新提交）。

use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};

use axum::{
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use uuid::Uuid;

use gateway_core::operation::ImageRequestKind;

/// 同时在跑的任务上限；排队任务不计入
const CONCURRENCY: usize = 4;
/// 运行中/排队任务总数上限，超过即拒绝提交
const MAX_ACTIVE_TASKS: usize = 64;
/// 终态任务保留时长（可被提前轮询到期）
const RETENTION: Duration = Duration::from_secs(24 * 3600);
/// 单次任务最长执行时间
const EXECUTION_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaskStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
}

impl TaskStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

struct TaskEntry {
    task_id: Uuid,
    /// 提交该任务的 Client Key 身份；轮询按它隔离
    owner: String,
    kind: ImageRequestKind,
    /// 请求体里的模型名（若解析得出），仅用于展示
    model: Option<String>,
    status: TaskStatus,
    created_at: DateTime<Utc>,
    finished_instant: Option<Instant>,
    /// 终态结果：上游响应原文 + 状态码
    result_body: Option<Bytes>,
    result_status: Option<u16>,
    error: Option<String>,
}

impl TaskEntry {
    fn public_view(&self) -> Value {
        let mut view = json!({
            "task_id": self.task_id.to_string(),
            "status": self.status.as_str(),
            "kind": match self.kind {
                ImageRequestKind::Generation => "generation",
                ImageRequestKind::Edit => "edit",
            },
            "created_at": self.created_at.to_rfc3339(),
            "poll_url": format!("/v1/images/tasks/{}", self.task_id),
        });
        if let Some(model) = &self.model {
            view["model"] = Value::String(model.clone());
        }
        match self.status {
            TaskStatus::Succeeded => {
                view["status_code"] = json!(self.result_status.unwrap_or(200));
                if let Some(body) = &self.result_body
                    && let Ok(parsed) = serde_json::from_slice::<Value>(body)
                {
                    view["result"] = parsed;
                }
            }
            TaskStatus::Failed => {
                view["status_code"] = json!(self.result_status.unwrap_or(502));
                view["error"] = json!(
                    self.error
                        .clone()
                        .unwrap_or_else(|| "image task failed".to_owned())
                );
                if let Some(body) = &self.result_body
                    && let Ok(parsed) = serde_json::from_slice::<Value>(body)
                {
                    view["detail"] = parsed;
                }
            }
            _ => {}
        }
        view
    }
}

#[derive(Default)]
struct Registry {
    tasks: HashMap<Uuid, TaskEntry>,
}

/// 异步生图任务登记表；提交接口持有它，执行协程回写它
#[derive(Clone)]
pub(crate) struct ImageTasks {
    registry: Arc<Mutex<Registry>>,
    semaphore: Arc<Semaphore>,
}

impl ImageTasks {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::default())),
            semaphore: Arc::new(Semaphore::new(CONCURRENCY)),
        }
    }

    fn registry(&self) -> MutexGuard<'_, Registry> {
        let mut registry = self
            .registry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        registry.tasks.retain(|_, entry| {
            entry
                .finished_instant
                .is_none_or(|at| at.elapsed() < RETENTION)
        });
        registry
    }

    /// 登记一个待执行任务，返回 task_id；在飞任务已达上限时返回 None
    fn register(
        &self,
        owner: String,
        kind: ImageRequestKind,
        model: Option<String>,
    ) -> Option<Uuid> {
        let mut registry = self.registry();
        let active = registry
            .tasks
            .values()
            .filter(|entry| matches!(entry.status, TaskStatus::Queued | TaskStatus::Running))
            .count();
        if active >= MAX_ACTIVE_TASKS {
            return None;
        }
        let task_id = Uuid::now_v7();
        registry.tasks.insert(
            task_id,
            TaskEntry {
                task_id,
                owner,
                kind,
                model,
                status: TaskStatus::Queued,
                created_at: Utc::now(),
                finished_instant: None,
                result_body: None,
                result_status: None,
                error: None,
            },
        );
        Some(task_id)
    }

    fn set_running(&self, task_id: Uuid) {
        if let Some(entry) = self.registry().tasks.get_mut(&task_id)
            && entry.status == TaskStatus::Queued
        {
            entry.status = TaskStatus::Running;
        }
    }

    fn finish(&self, task_id: Uuid, status: u16, body: Bytes) {
        if let Some(entry) = self.registry().tasks.get_mut(&task_id) {
            entry.status = if (200..300).contains(&status) {
                TaskStatus::Succeeded
            } else {
                TaskStatus::Failed
            };
            entry.result_status = Some(status);
            if entry.status == TaskStatus::Failed {
                entry.error = Some(format!("upstream returned status {status}"));
            }
            entry.result_body = Some(body);
            entry.finished_instant = Some(Instant::now());
        }
    }

    fn fail(&self, task_id: Uuid, error: String) {
        if let Some(entry) = self.registry().tasks.get_mut(&task_id) {
            entry.status = TaskStatus::Failed;
            entry.error = Some(error);
            entry.finished_instant = Some(Instant::now());
        }
    }

    /// 轮询任务；只能看到提交 Key 自己的任务
    pub(crate) fn get(&self, owner: &str, task_id: Uuid) -> Option<Value> {
        let registry = self.registry();
        let entry = registry
            .tasks
            .get(&task_id)
            .filter(|entry| entry.owner == owner)?;
        Some(entry.public_view())
    }
}

/// 提交入口：鉴权 + 登记 + spawn 执行；返回 202 + task_id + poll_url
///
/// `run` 由调用处构造：与同步 Images 请求完全相同的执行 future
/// （`provider_endpoint_response`），产出 axum `Response`。
pub(crate) fn submit<F, Fut>(
    tasks: &ImageTasks,
    owner: String,
    kind: ImageRequestKind,
    model: Option<String>,
    run: F,
) -> Response
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Response> + Send + 'static,
{
    let Some(task_id) = tasks.register(owner, kind, model) else {
        return task_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "too many image tasks in flight; retry later",
        );
    };
    let tasks = tasks.clone();
    tokio::spawn(async move {
        let _permit = match tasks.semaphore.acquire().await {
            Ok(permit) => permit,
            Err(_) => return, // semaphore never closed; unreachable
        };
        tasks.set_running(task_id);
        let outcome = tokio::time::timeout(EXECUTION_TIMEOUT, run()).await;
        match outcome {
            Ok(response) => {
                let status = response.status().as_u16();
                let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        tasks.fail(task_id, "failed to read upstream response".to_owned());
                        return;
                    }
                };
                tasks.finish(task_id, status, body);
            }
            Err(_) => tasks.fail(task_id, "image task timed out".to_owned()),
        }
    });
    (
        StatusCode::ACCEPTED,
        axum::Json(json!({
            "task_id": task_id.to_string(),
            "status": "queued",
            "poll_url": format!("/v1/images/tasks/{task_id}"),
            "created_at": Utc::now().to_rfc3339(),
        })),
    )
        .into_response()
}

pub(crate) fn task_error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        axum::Json(json!({
            "error": {
                "type": kind,
                "code": kind,
                "message": message,
            }
        })),
    )
        .into_response()
}

/// 从原始 JSON 请求体提取 model 字段（仅展示用；调度仍由 Provider 按正文解析）
pub(crate) fn extract_model(body: &Bytes) -> Option<String> {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("model")?.as_str().map(str::to_owned))
}

/// 检测客户端是否要求异步：`background: true` 出现在请求体顶层时转任务
pub(crate) fn wants_async(body: &Bytes) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("background")?.as_bool())
        .unwrap_or(false)
}

/// 剥除请求体顶层的 `background` 字段。
///
/// 异步语义由本网关的任务层承接；透传给上游会让上游返回自己的后台句柄，
/// 客户端拿不到任何可轮询的网关任务。preserve_order + arbitrary_precision
/// 保证其余字段原样往返。
pub(crate) fn strip_background(body: &Bytes) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(body) else {
        return body.clone();
    };
    let Some(object) = value.as_object_mut() else {
        return body.clone();
    };
    if object.remove("background").is_none() {
        return body.clone();
    }
    match serde_json::to_vec(&value) {
        Ok(bytes) => Bytes::from(bytes),
        Err(_) => body.clone(),
    }
}
