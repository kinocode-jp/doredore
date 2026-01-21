use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};
use tracing::{info, warn};

use doredore_core::core::enricher::Doredore;
use doredore_core::SearchMode;

// ============================================================================
// Application State
// ============================================================================

#[derive(Clone)]
struct AppState {
    rag: Arc<Doredore>,
}

// ============================================================================
// API Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
struct CreateCollectionRequest {
    name: String,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AddDocumentRequest {
    content: String,
    collection: Option<String>,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    q: String,
    collection: Option<String>,
    top_k: Option<usize>,
    threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct EnrichQuery {
    q: String,
    collection: Option<String>,
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ImportCsvRequest {
    file_path: String,
    collection: Option<String>,
    content_column: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message),
        }
    }
}

// ============================================================================
// API Handlers
// ============================================================================

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "doredore-server",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// List all collections
async fn list_collections(State(state): State<AppState>) -> impl IntoResponse {
    let rag = &state.rag;
    match rag.list_collections() {
        Ok(collections) => {
            let collections_data: Vec<_> = collections
                .into_iter()
                .map(|c| {
                    serde_json::json!({
                        "id": c.id,
                        "name": c.name,
                        "description": c.description,
                        "created_at": c.created_at
                    })
                })
                .collect();

            (StatusCode::OK, Json(ApiResponse::success(collections_data)))
        }
        Err(e) => {
            warn!("Failed to list collections: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Create a new collection
async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    let rag = &state.rag;
    match rag.create_collection(&req.name, req.description.as_deref()) {
        Ok(id) => {
            info!("Created collection '{}' with id {}", req.name, id);
            (
                StatusCode::CREATED,
                Json(ApiResponse::success(serde_json::json!({
                    "id": id,
                    "name": req.name
                }))),
            )
        }
        Err(e) => {
            warn!("Failed to create collection: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Delete a collection
async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let rag = &state.rag;
    match rag.delete_collection(&name) {
        Ok(_) => {
            info!("Deleted collection '{}'", name);
            (
                StatusCode::OK,
                Json(ApiResponse::success(serde_json::json!({
                    "message": format!("Collection '{}' deleted", name)
                }))),
            )
        }
        Err(e) => {
            warn!("Failed to delete collection: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Add a document
async fn add_document(
    State(state): State<AppState>,
    Json(req): Json<AddDocumentRequest>,
) -> impl IntoResponse {
    let collection = req.collection.as_deref().unwrap_or("default");

    let rag = &state.rag;
    match rag.add_document(&req.content, collection, req.metadata.as_ref()) {
        Ok(id) => {
            info!("Added document {} to collection '{}'", id, collection);
            (
                StatusCode::CREATED,
                Json(ApiResponse::success(serde_json::json!({
                    "id": id,
                    "collection": collection
                }))),
            )
        }
        Err(e) => {
            warn!("Failed to add document: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Delete a document
async fn delete_document(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> impl IntoResponse {
    let rag = &state.rag;
    match rag.delete_document(id) {
        Ok(_) => {
            info!("Deleted document {}", id);
            (
                StatusCode::OK,
                Json(ApiResponse::success(serde_json::json!({
                    "message": format!("Document {} deleted", id)
                }))),
            )
        }
        Err(e) => {
            warn!("Failed to delete document: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// List documents
async fn list_documents(
    State(state): State<AppState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let collection = params.get("collection").map(|s| s.as_str());
    let limit = params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let offset = params
        .get("offset")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let rag = &state.rag;
    match rag.list_documents(collection, limit, offset) {
        Ok(documents) => {
            let docs_data: Vec<_> = documents
                .into_iter()
                .map(|d| {
                    serde_json::json!({
                        "id": d.id,
                        "collection_id": d.collection_id,
                        "content": d.content,
                        "metadata": d.metadata,
                        "created_at": d.created_at
                    })
                })
                .collect();

            (StatusCode::OK, Json(ApiResponse::success(docs_data)))
        }
        Err(e) => {
            warn!("Failed to list documents: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Search for similar documents
async fn search(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> impl IntoResponse {
    let top_k = query.top_k.unwrap_or(5);
    let threshold = query.threshold.unwrap_or(0.0);

    let rag = &state.rag;
    match rag.search(&query.q, query.collection.as_deref(), None, top_k, threshold, SearchMode::Semantic, None) {
        Ok(results) => {
            let results_data: Vec<_> = results
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "document_id": r.document_id,
                        "content": r.content,
                        "score": r.score,
                        "collection": r.collection_name,
                        "metadata": r.metadata
                    })
                })
                .collect();

            (
                StatusCode::OK,
                Json(ApiResponse::success(serde_json::json!({
                    "query": query.q,
                    "results": results_data,
                    "count": results_data.len()
                }))),
            )
        }
        Err(e) => {
            warn!("Search failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Enrich query with context (main RAG function)
async fn enrich(
    State(state): State<AppState>,
    Query(query): Query<EnrichQuery>,
) -> impl IntoResponse {
    let top_k = query.top_k.unwrap_or(3);

    let rag = &state.rag;
    match rag.enrich(&query.q, query.collection.as_deref(), None, top_k, 0.0, SearchMode::Semantic, None) {
        Ok(result) => {
            let sources: Vec<_> = result
                .sources
                .into_iter()
                .map(|s| {
                    serde_json::json!({
                        "document_id": s.document_id,
                        "content": s.content,
                        "score": s.score,
                        "collection": s.collection_name,
                        "metadata": s.metadata
                    })
                })
                .collect();

            (
                StatusCode::OK,
                Json(ApiResponse::success(serde_json::json!({
                    "query": result.question,
                    "context": result.context,
                    "sources": sources,
                    "source_count": sources.len()
                }))),
            )
        }
        Err(e) => {
            warn!("Enrich failed: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Import CSV
async fn import_csv(
    State(state): State<AppState>,
    Json(req): Json<ImportCsvRequest>,
) -> impl IntoResponse {
    let collection = req.collection.as_deref().unwrap_or("default");
    let content_column = req.content_column.as_deref().unwrap_or("content");

    let rag = &state.rag;
    match rag.import_csv(&req.file_path, collection, content_column, None) {
        Ok(count) => {
            info!("Imported {} documents from {}", count, req.file_path);
            (
                StatusCode::OK,
                Json(ApiResponse::success(serde_json::json!({
                    "count": count,
                    "collection": collection
                }))),
            )
        }
        Err(e) => {
            warn!("CSV import failed: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::error(e.to_string())),
            )
        }
    }
}

/// Serve admin UI
async fn admin_ui() -> impl IntoResponse {
    Html(include_str!("../static/index.html"))
}

// ============================================================================
// Main Application
// ============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Get configuration from environment
    let db_path = std::env::var("DATABASE_PATH").unwrap_or_else(|_| "./knowledge.db".to_string());
    let model = std::env::var("EMBEDDING_MODEL").unwrap_or_else(|_| "bge-small-en-v1.5".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    info!("Initializing Doredore...");
    let rag = Doredore::new(&db_path, Some(&model), None)?;
    info!("Doredore initialized with model: {}", model);

    let state = AppState {
        rag: Arc::new(rag),
    };

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build API routes
    let api_routes = Router::new()
        // Collections
        .route("/collections", get(list_collections).post(create_collection))
        .route("/collections/:name", delete(delete_collection))
        // Documents
        .route("/documents", get(list_documents).post(add_document))
        .route("/documents/:id", delete(delete_document))
        // Search & Enrich
        .route("/search", get(search))
        .route("/enrich", get(enrich))
        // CSV
        .route("/import-csv", post(import_csv))
        .with_state(state.clone());

    // Build main app
    let app = Router::new()
        .route("/", get(admin_ui))
        .route("/health", get(health_check))
        .nest("/api", api_routes)
        .nest_service("/static", ServeDir::new("static"))
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("");
    info!("==========================================================");
    info!("  doredore Server");
    info!("==========================================================");
    info!("");
    info!("🚀 Server running on http://{}", addr);
    info!("");
    info!("API Endpoints:");
    info!("  GET    /health");
    info!("  GET    /api/collections");
    info!("  POST   /api/collections");
    info!("  DELETE /api/collections/:name");
    info!("  GET    /api/documents");
    info!("  POST   /api/documents");
    info!("  DELETE /api/documents/:id");
    info!("  GET    /api/search?q=...");
    info!("  GET    /api/enrich?q=...");
    info!("  POST   /api/import-csv");
    info!("");
    info!("Admin UI:");
    info!("  http://{}/", addr);
    info!("");
    info!("==========================================================");

    axum::serve(listener, app).await?;

    Ok(())
}
