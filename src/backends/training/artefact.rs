//! Training output artefact.
use serde::{Deserialize, Serialize};

use crate::backends::types::{AdapterRef, ModelId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterArtifact {
    pub adapter: AdapterRef,
    pub base_model: ModelId,
    pub steps: u32,
    pub final_loss: f32,
    pub training_seconds: u64,
}
