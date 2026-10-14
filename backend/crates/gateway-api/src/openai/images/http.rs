//! Codex Images 非流式 HTTP adapter

use std::net::SocketAddr;

use axum::extract::FromRequest;
use axum::{
    body::Bytes,
    extract::{Extension, State, connect_info::ConnectInfo},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use gateway_core::engine::execution::ClientTransport;
use gateway_core::error::{GatewayError, GatewayErrorKind};
use gateway_core::operation::{
    ImageRequest, ImageRequestKind, Operation, OperationKind, RawJsonPayload,
};
use serde_json::Value;
use uuid::Uuid;

use super::tasks;

use crate::ApiState;
use crate::middleware::headers::encode_headers;
use crate::openai::middleware::RequestInput;
use crate::openai::{
    auth::{authenticate_client, client_access_error_response},
    endpoint::provider_endpoint_response,
    responses::{OpenAiRequestHeaders, request_client_context},
    router::{IMAGE_EDITS_PATH, IMAGE_GENERATIONS_PATH},
};

const OPENAI_PROTOCOL: &str = "openai";
const IMAGE_TURN_ID_CONTEXT_KEY: &str = "image_turn_id";

/// `POST /v1/images/generations`
///
/// `background: true` 的请求直接走异步任务通道，与显式 tasks 端点一致
pub(crate) async fn image_generations(
    State(state): State<ApiState>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if tasks::wants_async(&body) {
        return submit_image_task(
            state,
            connect_info,
            headers,
            body,
            ImageRequestKind::Generation,
            IMAGE_GENERATIONS_PATH,
        )
        .await;
    }
    handle_image_request(
        state,
        connect_info,
        headers,
        body,
        ImageRequestKind::Generation,
        IMAGE_GENERATIONS_PATH,
    )
    .await
}

/// `POST /v1/images/edits`
pub(crate) async fn image_edits(
    State(state): State<ApiState>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle_image_request(
        state,
        connect_info,
        headers,
        body,
        ImageRequestKind::Edit,
        IMAGE_EDITS_PATH,
    )
    .await
}

async fn handle_image_request(
    state: ApiState,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
    kind: ImageRequestKind,
    endpoint: &'static str,
) -> Response {
    let service = state.openai();
    let client = match authenticate_client(service, &headers).await {
        Ok(client) => client,
        Err(error) => return client_access_error_response(error),
    };
    let (client_ip, user_agent) = request_client_context(
        &headers,
        connect_info.map(|Extension(ConnectInfo(address))| address),
    );
    provider_endpoint_response(
        service.clone(),
        client,
        RequestInput {
            endpoint: endpoint.to_owned(),
            protocol: OPENAI_PROTOCOL.to_owned(),
            operation: Some(OperationKind::GenerateImage),
            transport: ClientTransport::HttpJson,
            model_hint: None,
            headers: encode_headers(&headers),
            body,
        },
        client_ip,
        user_agent,
        move |body, headers| image_operation(body, headers, kind),
    )
    .await
}

/// `POST /v1/images/generations/tasks`
///
/// 与同步 generations 相同入参，登记后台任务并立即返回 202 + poll_url。
/// 客户端随后用 `GET /v1/images/tasks/{task_id}` 轮询。
pub(crate) async fn image_generations_task(
    State(state): State<ApiState>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    submit_image_task(
        state,
        connect_info,
        headers,
        body,
        ImageRequestKind::Generation,
        IMAGE_GENERATIONS_PATH,
    )
    .await
}

/// `POST /v1/images/edits/tasks`
pub(crate) async fn image_edits_task(
    State(state): State<ApiState>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    submit_image_task(
        state,
        connect_info,
        headers,
        body,
        ImageRequestKind::Edit,
        IMAGE_EDITS_PATH,
    )
    .await
}

async fn submit_image_task(
    state: ApiState,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
    kind: ImageRequestKind,
    endpoint: &'static str,
) -> Response {
    let service = state.openai();
    let client = match authenticate_client(service, &headers).await {
        Ok(client) => client,
        Err(error) => return client_access_error_response(error),
    };
    let owner = client.policy().key_id().as_str().to_owned();
    let (client_ip, user_agent) = request_client_context(
        &headers,
        connect_info.map(|Extension(ConnectInfo(address))| address),
    );
    // 代理接管异步语义：上游一律收到同步请求
    let body = tasks::strip_background(&body);
    let model = tasks::extract_model(&body);
    let service = service.clone();
    tasks::submit(
        state.image_tasks(),
        owner,
        kind,
        model,
        move || async move {
            provider_endpoint_response(
                service,
                client,
                RequestInput {
                    endpoint: endpoint.to_owned(),
                    protocol: OPENAI_PROTOCOL.to_owned(),
                    operation: Some(OperationKind::GenerateImage),
                    transport: ClientTransport::HttpJson,
                    model_hint: None,
                    headers: encode_headers(&headers),
                    body,
                },
                client_ip,
                user_agent,
                move |body, headers| image_operation(body, headers, kind),
            )
            .await
        },
    )
}

fn image_operation(
    body: Bytes,
    headers: &HeaderMap,
    kind: ImageRequestKind,
) -> Result<Operation, GatewayError> {
    let mut context = OpenAiRequestHeaders::from_headers(headers).session_context();
    if let Some(turn_id) = headers
        .get("x-codex-image-turn-id")
        .and_then(|value| value.to_str().ok())
    {
        context.insert(
            IMAGE_TURN_ID_CONTEXT_KEY.to_owned(),
            Value::String(turn_id.to_owned()),
        );
    }
    let payload = RawJsonPayload::new(OPENAI_PROTOCOL, body)
        .map_err(|_| {
            GatewayError::new(
                GatewayErrorKind::Internal,
                "OpenAI protocol identifier is invalid",
            )
        })?
        .with_context(context);
    Ok(Operation::GenerateImage(ImageRequest::from_raw_json(
        kind, payload,
    )))
}

/// 异步生图兼容契约：
/// 提交返回 202 + `task_id`/`object`/`poll_url`，并带 `Location`/`Retry-After`/`Cache-Control`；
/// 任务状态值为 processing/completed/failed，`error` 为 `{type,message}` 对象。
const IMAGE_TASK_TTL_SECONDS: i64 = 24 * 3600;

/// `POST /images/generations/async`
pub(crate) async fn image_generations_async(
    State(state): State<ApiState>,
    uri: axum::http::Uri,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let response = submit_image_task(
        state,
        connect_info,
        headers,
        body,
        ImageRequestKind::Generation,
        IMAGE_GENERATIONS_PATH,
    )
    .await;
    async_submit_response(response, uri.path()).await
}

/// `POST /images/edits/async`（multipart 转 JSON 透传；上游 edits 只收 JSON）
pub(crate) async fn image_edits_async(
    State(state): State<ApiState>,
    uri: axum::http::Uri,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let (headers, body) = match edits_multipart_to_json(headers, body).await {
        Ok(parts) => parts,
        Err(response) => return response,
    };
    let response = submit_image_task(
        state,
        connect_info,
        headers,
        body,
        ImageRequestKind::Edit,
        IMAGE_EDITS_PATH,
    )
    .await;
    async_submit_response(response, uri.path()).await
}

/// 兼容客户端的 edits/async 统一收 multipart；上游 `/codex/images/edits` 只认
/// JSON（`images[].image_url` / `mask.image_url`），这里做一次性转换。
/// 非 multipart 请求原样放行。
async fn edits_multipart_to_json(
    mut headers: HeaderMap,
    body: Bytes,
) -> Result<(HeaderMap, Bytes), Response> {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    if !content_type
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
    {
        return Ok((headers, body));
    }
    let mut multipart = axum::extract::Multipart::from_request(
        axum::http::Request::builder()
            .header(axum::http::header::CONTENT_TYPE, &content_type)
            .body(axum::body::Body::from(body))
            .map_err(|_| {
                tasks::task_error(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "failed to construct multipart request",
                )
            })?,
        &(),
    )
    .await
    .map_err(|_| {
        tasks::task_error(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid multipart body",
        )
    })?;

    const MAX_PART_BYTES: usize = 32 * 1024 * 1024;
    const MAX_FILE_PARTS: usize = 16;
    // 透传的文本字段（model 必填，其余可选）
    const TEXT_FIELDS: &[&str] = &[
        "model",
        "prompt",
        "size",
        "quality",
        "n",
        "background",
        "output_format",
        "moderation",
        "output_compression",
        "response_format",
        "style",
        "input_fidelity",
        "user",
    ];
    let mut fields = serde_json::Map::new();
    let mut images: Vec<Value> = Vec::new();
    let mut mask_url: Option<String> = None;

    while let Some(field) = multipart.next_field().await.map_err(|_| {
        tasks::task_error(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "failed to read multipart field",
        )
    })? {
        let name = field.name().unwrap_or_default().to_owned();
        let file_name = field.file_name().map(str::to_owned);
        let part_content_type = field.content_type().map(str::to_owned);
        if file_name.is_some() {
            // 文件字段：image / image[] → images[].image_url；mask → mask.image_url
            if images.len() >= MAX_FILE_PARTS {
                return Err(tasks::task_error(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "too many image files",
                ));
            }
            let data = field.bytes().await.map_err(|_| {
                tasks::task_error(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "failed to read uploaded file",
                )
            })?;
            if data.len() > MAX_PART_BYTES {
                return Err(tasks::task_error(
                    axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                    "invalid_request_error",
                    "uploaded file exceeds size limit",
                ));
            }
            if data.is_empty() {
                continue;
            }
            let mime = part_content_type
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| {
                    sniff_image_mime(&data)
                        .unwrap_or("application/octet-stream")
                        .to_owned()
                });
            if !mime.to_ascii_lowercase().starts_with("image/") {
                return Err(tasks::task_error(
                    axum::http::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "uploaded file must be an image",
                ));
            }
            let data_url = format!(
                "data:{mime};base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&data)
            );
            if name == "mask" {
                mask_url = Some(data_url);
            } else if name == "image" || name.starts_with("image[") {
                images.push(serde_json::json!({"image_url": data_url}));
            }
            continue;
        }
        // 文本字段
        let value = field.text().await.map_err(|_| {
            tasks::task_error(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "failed to read form field",
            )
        })?;
        if !TEXT_FIELDS.contains(&name.as_str()) {
            continue;
        }
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        fields.insert(
            name.clone(),
            match name.as_str() {
                "n" | "output_compression" => value
                    .parse::<u64>()
                    .map(serde_json::Value::from)
                    .unwrap_or_else(|_| Value::String(value.to_owned())),
                _ => Value::String(value.to_owned()),
            },
        );
    }

    if !images.is_empty() {
        fields.insert("images".to_owned(), Value::Array(images));
    }
    if let Some(mask_url) = mask_url {
        fields.insert(
            "mask".to_owned(),
            serde_json::json!({"image_url": mask_url}),
        );
    }
    let json_body = serde_json::to_vec(&Value::Object(fields)).map_err(|_| {
        tasks::task_error(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "failed to encode image request",
        )
    })?;
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    headers.remove(axum::http::header::CONTENT_LENGTH);
    Ok((headers, Bytes::from(json_body)))
}

/// `GET /images/tasks/{task_id}`
pub(crate) async fn image_task_status(
    State(state): State<ApiState>,
    headers: HeaderMap,
    axum::extract::Path(task_id): axum::extract::Path<String>,
) -> Response {
    let service = state.openai();
    let client = match authenticate_client(service, &headers).await {
        Ok(client) => client,
        Err(error) => return client_access_error_response(error),
    };
    let task_id = match Uuid::parse_str(&task_id) {
        Ok(id) => id,
        Err(_) => {
            return tasks::task_error(
                axum::http::StatusCode::NOT_FOUND,
                "not_found_error",
                "image task not found",
            );
        }
    };
    match state
        .image_tasks()
        .get(client.policy().key_id().as_str(), task_id)
    {
        Some(view) => async_task_view(&view).into_response(),
        None => tasks::task_error(
            axum::http::StatusCode::NOT_FOUND,
            "not_found_error",
            "image task not found or expired",
        ),
    }
}

/// 把任务登记响应改写成异步 submit 契约
async fn async_submit_response(response: Response, submit_path: &str) -> Response {
    if response.status() != axum::http::StatusCode::ACCEPTED {
        return response;
    }
    let body = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return tasks::task_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "failed to read task submission",
            );
        }
    };
    let view: Value = match serde_json::from_slice(&body) {
        Ok(view) => view,
        Err(_) => {
            return tasks::task_error(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "failed to read task submission",
            );
        }
    };
    let task_id = view
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let created_at = chrono::Utc::now().timestamp();
    let poll_url = if submit_path.starts_with("/v1/") {
        format!("/v1/images/tasks/{task_id}")
    } else {
        format!("/images/tasks/{task_id}")
    };
    let mut response = (
        axum::http::StatusCode::ACCEPTED,
        [
            ("cache-control", "no-store"),
            ("location", poll_url.as_str()),
            ("retry-after", "3"),
        ],
        axum::Json(serde_json::json!({
            "id": task_id,
            "task_id": task_id,
            "object": "image.generation.task",
            "status": "processing",
            "created_at": created_at,
            "expires_at": created_at + IMAGE_TASK_TTL_SECONDS,
            "poll_url": poll_url,
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert("cache-control", "no-store".parse().unwrap());
    response
}

/// 任务公开视图：processing/completed/failed + error{type,message}
fn async_task_view(view: &Value) -> Response {
    let task_id = view
        .get("task_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let status = match view.get("status").and_then(Value::as_str) {
        Some("succeeded") => "completed",
        Some("failed") => "failed",
        _ => "processing",
    };
    let created_at = chrono::DateTime::parse_from_rfc3339(
        view.get("created_at").and_then(Value::as_str).unwrap_or(""),
    )
    .map(|ts| ts.timestamp())
    .unwrap_or_else(|_| chrono::Utc::now().timestamp());
    let mut out = serde_json::json!({
        "id": task_id,
        "task_id": task_id,
        "object": "image.generation.task",
        "status": status,
        "created_at": created_at,
        "expires_at": created_at + IMAGE_TASK_TTL_SECONDS,
    });
    if let Some(code) = view.get("status_code").and_then(Value::as_u64) {
        out["http_status"] = serde_json::json!(code);
    }
    if status == "completed" {
        if let Some(result) = view.get("result") {
            out["result"] = result.clone();
            if let Some(url) = result
                .get("data")
                .and_then(Value::as_array)
                .and_then(|data| data.first())
                .and_then(|item| item.get("url"))
                .and_then(Value::as_str)
            {
                out["image_url"] = Value::String(url.to_owned());
            }
        }
        out["completed_at"] = serde_json::json!(chrono::Utc::now().timestamp());
    } else if status == "failed" {
        out["completed_at"] = serde_json::json!(chrono::Utc::now().timestamp());
        let error = view
            .get("detail")
            .cloned()
            .map(|detail| {
                // 公开契约存的是内层 error 对象；上游错误体是 {"error":{...}} 时拆一层
                match detail.get("error") {
                    Some(e) if e.is_object() => e.clone(),
                    _ => detail,
                }
            })
            .unwrap_or_else(|| {
                serde_json::json!({
                    "type": "api_error",
                    "message": view
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("image generation failed"),
                })
            });
        out["error"] = error;
    }
    axum::Json(out).into_response()
}

/// 常见图片格式的魔数嗅探；上游要求 data URL 的 image/* 前缀
fn sniff_image_mime(data: &[u8]) -> Option<&'static str> {
    const PNG: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    const JPEG: [u8; 3] = [0xFF, 0xD8, 0xFF];
    if data.starts_with(&PNG) {
        Some("image/png")
    } else if data.starts_with(&JPEG) {
        Some("image/jpeg")
    } else if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if data.len() > 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}
