/// Configuration loading with 3-tier precedence:
///   1. Compiled defaults
///   2. TOML config file (~/.config/writer/config.toml)
///   3. Environment variables (WRITER_*)
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::AppError;

// ── Config structs ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub active_profile: String,
    pub base_model: String,
    pub update: UpdateConfig,
    pub inference: InferenceConfig,
    pub decoding: DecodingConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateConfig {
    pub enabled: bool,
    pub owner: String,
    pub repo: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub backend: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub ollama_url: String,
    /// Prompt mode for inference: "chat" (default) or "raw".
    /// In raw mode, mlx_generate.py bypasses chat template and sends prompt verbatim.
    pub prompt_mode: PromptMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingConfig {
    pub n_candidates: u16,
    pub max_tokens: u32,
    /// Maximum generation attempts before returning best candidate.
    /// None = default (3). Set to 1 for ablation runs.
    #[serde(default)]
    pub max_attempts: Option<u16>,
    /// Contrastive alpha for CoPe-style decoding. Set to 0.0 to disable.
    /// Requires a contrastive_base model to be specified in generation request.
    /// Currently only supported by Ollama backend.
    pub contrastive_alpha: f32,
    pub banned_word_bias: f32,
    pub preferred_word_bias: f32,
    pub kv_quant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub backend: String,
    pub rank: u16,
    pub alpha: f32,
    pub learning_rate: f32,
    pub batch_size: u16,
    pub max_steps: u32,
    pub max_seq_len: u32,
    /// Dataset format for LoRA training: "chat", "completions", or "text".
    /// - chat: {"messages": [{role, content}, ...]} — current default
    /// - completions: {"prompt": "...", "completion": "..."} — supports mask_prompt
    /// - text: {"text": "..."} — raw continuation, fully custom formatting
    pub dataset_format: DatasetFormat,
    /// When true and dataset_format is completions or chat, mask the prompt
    /// tokens so the model only learns from the completion/assistant turn.
    pub mask_prompt: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    #[default]
    Chat,
    Completions,
    Text,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    #[default]
    Chat,
    Raw,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            active_profile: "default".into(),
            base_model: "google/gemma-4-26b-a4b".into(),
            update: UpdateConfig::default(),
            inference: InferenceConfig::default(),
            decoding: DecodingConfig::default(),
            training: TrainingConfig::default(),
        }
    }
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            owner: "199-biotechnologies".into(),
            repo: "writer".into(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: "ollama".into(),
            temperature: 0.7,
            max_tokens: 2048,
            ollama_url: "http://localhost:11434".into(),
            prompt_mode: PromptMode::default(),
        }
    }
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            n_candidates: 8,
            max_tokens: 4096,
            max_attempts: None,
            contrastive_alpha: 0.0, // disabled until contrastive_base model is wired
            banned_word_bias: -4.0,
            preferred_word_bias: 1.5,
            kv_quant: "auto".into(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: "mlx-tune".into(),
            rank: 16,
            alpha: 32.0,
            // Conservative defaults for large models (26B).
            // Gemma 4 26B: batch=1, seq=2048, lr=2e-5.
            // Smaller models can use batch=4, lr=1e-4.
            learning_rate: 2e-5,
            batch_size: 1,
            max_steps: 500,
            max_seq_len: 2048,
            dataset_format: DatasetFormat::default(),
            mask_prompt: false,
        }
    }
}

// ── Paths ──────────────────────────────────────────────────────────────────

pub fn config_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "writer")
        .map(|d| d.config_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn data_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "writer")
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn profiles_dir() -> PathBuf {
    data_dir().join("profiles")
}

pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

// ── Loading ────────────────────────────────────────────────────────────────

pub fn load() -> Result<AppConfig, AppError> {
    use figment::Figment;
    use figment::providers::{Env, Format as _, Serialized, Toml};

    Figment::from(Serialized::defaults(AppConfig::default()))
        .merge(Toml::file(config_path()))
        .merge(Env::prefixed("WRITER_").split("_"))
        .extract()
        .map_err(|e| AppError::Config(e.to_string()))
}
