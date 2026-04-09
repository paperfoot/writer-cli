//! Training backends (mlx-tune, unsloth, ...).
pub mod artefact;
pub mod config;

use async_trait::async_trait;

use self::artefact::AdapterArtifact;
use self::config::{DpoConfig, LoraConfig, TrainingProgress};

#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("training backend unavailable: {0}")]
    Unavailable(String),

    #[error("dataset error: {0}")]
    Dataset(String),

    #[error("training failed: {0}")]
    TrainingFailed(String),

    #[error("not implemented yet")]
    NotImplemented,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[async_trait]
pub trait TrainingBackend: Send + Sync {
    fn name(&self) -> &str;

    async fn train_lora(
        &self,
        config: LoraConfig,
        on_progress: &(dyn Fn(TrainingProgress) + Sync),
    ) -> Result<AdapterArtifact, TrainingError>;

    async fn train_dpo(
        &self,
        config: DpoConfig,
        on_progress: &(dyn Fn(TrainingProgress) + Sync),
    ) -> Result<AdapterArtifact, TrainingError>;
}
