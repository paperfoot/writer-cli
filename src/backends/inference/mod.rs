//! Inference backends (Ollama, mistral.rs, custom MLX, ...).
pub mod capabilities;
pub mod mlx;
pub mod mlx_worker;
pub mod ollama;
pub mod request;
pub mod response;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use self::capabilities::BackendCapabilities;
use self::request::GenerationRequest;
use self::response::GenerationEvent;
use crate::backends::types::{ModelHandle, ModelId};

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("backend unavailable: {0}")]
    Unavailable(String),

    #[error("model not found: {0}")]
    ModelNotFound(ModelId),

    #[error("adapter not found: {0}")]
    AdapterNotFound(String),

    #[error("capability not supported: {0}")]
    CapabilityNotSupported(&'static str),

    #[error("network error: {0}")]
    Network(String),

    #[error("backend returned error: {0}")]
    Backend(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListing {
    pub id: ModelId,
    pub is_downloaded: bool,
    pub size_bytes: Option<u64>,
}

/// A backend that can generate text from a prompt.
///
/// Backends MUST:
/// - report honest capabilities (never claim a feature they do not run)
/// - stream events in candidate-index order within a single candidate,
///   but events from different candidates may interleave
/// - return `BackendError::CapabilityNotSupported` rather than silently
///   falling back to a worse path
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> BackendCapabilities;

    async fn list_models(&self) -> Result<Vec<ModelListing>, BackendError>;
    async fn load_model(&self, id: &ModelId) -> Result<ModelHandle, BackendError>;

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Box<dyn Stream<Item = GenerationEvent> + Send + Unpin>, BackendError>;
}
