//! Training configuration vocabulary.
//!
//! All knobs are typed. Backends must accept or reject each field; silent
//! fallbacks for unsupported settings are a bug.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::backends::types::ModelId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub profile: String,
    pub base_model: ModelId,
    pub dataset_dir: PathBuf,
    pub adapter_out: PathBuf,
    pub rank: u16,
    pub alpha: f32,
    pub learning_rate: f32,
    pub batch_size: u16,
    pub max_steps: u32,
    pub max_seq_len: u32,
    /// When true, mask prompt tokens during training so the model
    /// only learns from the completion/assistant portion.
    pub mask_prompt: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreferenceMethod {
    #[default]
    Simpo,
    Dpo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    pub profile: String,
    pub base_model: ModelId,
    pub preference_dataset: PathBuf,
    pub adapter_out: PathBuf,
    /// Path to the SFT adapter to continue training from.
    pub base_adapter: Option<PathBuf>,
    /// Preference optimization method.
    pub method: PreferenceMethod,
    /// KL penalty coefficient (DPO) or reward margin (SimPO gamma).
    pub beta: f32,
    /// SimPO gamma — target reward margin between chosen and rejected.
    pub gamma: f32,
    pub learning_rate: f32,
    pub batch_size: u16,
    pub max_steps: u32,
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub step: u32,
    pub total_steps: u32,
    pub loss: f32,
    pub learning_rate: f32,
    pub tokens_per_second: f32,
}
