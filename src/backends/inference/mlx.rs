//! MLX direct inference backend — uses mlx_lm via Python subprocess.
//!
//! Used when a LoRA adapter is present, since Ollama (GGUF) can't load
//! safetensors adapters produced by mlx_lm.lora.
//!
//! Reference: mlx-lm (https://github.com/ml-explore/mlx-lm)
use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use super::capabilities::{BackendCapabilities, KvQuantKind};
use super::request::GenerationRequest;
use super::response::{FinishReason, GenerationEvent, UsageStats};
use super::{BackendError, InferenceBackend, ModelListing};
use crate::backends::types::{ModelHandle, ModelId};

pub struct MlxBackend {
    script_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct MlxRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_path: Option<String>,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_prompt: Option<String>,
    prompt_mode: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    logit_bias: std::collections::HashMap<String, f32>,
}

#[derive(Debug, Deserialize)]
struct MlxResponse {
    text: String,
    prompt_tokens: u32,
    generation_tokens: u32,
    #[allow(dead_code)]
    generation_tps: f64,
    #[allow(dead_code)]
    peak_memory_gb: f64,
    finish_reason: Option<String>,
    elapsed_ms: u64,
}

impl MlxBackend {
    pub fn new() -> Result<Self, BackendError> {
        // Locate the bridge script relative to the binary or via env
        let script = Self::find_script()?;
        Ok(Self {
            script_path: script,
        })
    }

    fn find_script() -> Result<PathBuf, BackendError> {
        // Check env override first
        if let Ok(p) = std::env::var("WRITER_MLX_SCRIPT") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Ok(path);
            }
        }

        // Check relative to the binary
        if let Ok(exe) = std::env::current_exe() {
            let candidates = [
                exe.parent()
                    .unwrap_or(Path::new("."))
                    .join("../scripts/mlx_generate.py"),
                exe.parent()
                    .unwrap_or(Path::new("."))
                    .join("../../scripts/mlx_generate.py"),
            ];
            for c in &candidates {
                if let Ok(canonical) = c.canonicalize() {
                    if canonical.exists() {
                        return Ok(canonical);
                    }
                }
            }
        }

        // Check in the source tree (dev mode)
        let dev_path = PathBuf::from("scripts/mlx_generate.py");
        if dev_path.exists() {
            return Ok(dev_path.canonicalize().unwrap_or(dev_path));
        }

        Err(BackendError::Unavailable(
            "scripts/mlx_generate.py not found. Set WRITER_MLX_SCRIPT env var.".into(),
        ))
    }

    /// Resolve the HuggingFace model ID for MLX.
    /// google/gemma-4-26b-a4b -> mlx-community/gemma-4-26b-a4b-it-4bit
    fn resolve_mlx_model(model_id: &ModelId) -> String {
        let name = model_id.name();
        // If already an mlx-community model, pass through
        if model_id.owner() == "mlx-community" {
            return format!("{}/{}", model_id.owner(), name);
        }
        // Map common models to their MLX quantized variants
        if name.contains("gemma-4-26b") || name.contains("gemma4-26b") {
            return "mlx-community/gemma-4-26b-a4b-it-4bit".to_string();
        }
        if name.contains("gemma-3-4b") || name.contains("gemma3-4b") {
            return "mlx-community/gemma-3-4b-it-4bit".to_string();
        }
        // Default: try mlx-community with -4bit suffix
        format!("mlx-community/{}-4bit", name)
    }

    async fn generate_single(
        &self,
        mlx_model: &str,
        adapter_path: Option<&Path>,
        request: &GenerationRequest,
        candidate_index: u16,
    ) -> Result<Vec<GenerationEvent>, BackendError> {
        let mlx_req = MlxRequest {
            model: mlx_model.to_string(),
            adapter_path: adapter_path.map(|p| p.to_string_lossy().to_string()),
            prompt: request.prompt.clone(),
            system_prompt: request.system_prompt.clone(),
            prompt_mode: request.prompt_mode.clone().unwrap_or_else(|| "chat".to_string()),
            max_tokens: request.params.max_tokens,
            temperature: request.params.temperature,
            top_p: request.params.top_p,
            repetition_penalty: request.params.repetition_penalty,
            logit_bias: request.logit_bias.clone(),
        };

        let input_json =
            serde_json::to_string(&mlx_req).map_err(|e| BackendError::Backend(e.to_string()))?;

        let mut child = tokio::process::Command::new("python3")
            .arg(&self.script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                BackendError::Unavailable(format!("Failed to spawn mlx_generate.py: {e}"))
            })?;

        // Write request to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin
                .write_all(input_json.as_bytes())
                .await
                .map_err(|e| BackendError::Backend(format!("Failed to write to stdin: {e}")))?;
            drop(stdin);
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| BackendError::Backend(format!("Process failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(BackendError::Backend(format!(
                "mlx_generate.py failed: {stderr}"
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let resp: MlxResponse = serde_json::from_str(&stdout).map_err(|e| {
            BackendError::Backend(format!(
                "Failed to parse mlx_generate.py output: {e}\nRaw: {stdout}"
            ))
        })?;

        let finish = match resp.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::MaxTokens,
            _ => FinishReason::Stop,
        };

        let events = vec![
            GenerationEvent::Token {
                candidate_index,
                text: resp.text.clone(),
                logprob: 0.0,
            },
            GenerationEvent::Done {
                candidate_index,
                finish_reason: finish,
                usage: UsageStats {
                    prompt_tokens: resp.prompt_tokens,
                    generated_tokens: resp.generation_tokens,
                    elapsed_ms: resp.elapsed_ms,
                },
                full_text: resp.text,
            },
        ];

        Ok(events)
    }
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    fn name(&self) -> &str {
        "mlx"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_lora: true,
            supports_logit_bias: true,
            supports_contrastive_decoding: false,
            supports_speculative_decoding: false,
            supports_activation_steering: false,
            kv_quant: KvQuantKind::None,
            quant_schemes: vec![],
            max_context: 8192,
            streaming: false,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelListing>, BackendError> {
        // MLX models are on-disk; listing HF cache is complex — return empty
        Ok(vec![])
    }

    async fn load_model(&self, id: &ModelId) -> Result<ModelHandle, BackendError> {
        let mlx_model = Self::resolve_mlx_model(id);
        Ok(ModelHandle(mlx_model))
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Box<dyn Stream<Item = GenerationEvent> + Send + Unpin>, BackendError> {
        let n = request.params.n_candidates as usize;
        let mlx_model = &handle.0;
        let adapter_path = request.adapter.as_ref().map(|a| a.path.as_path());

        if n <= 1 {
            let events = self
                .generate_single(mlx_model, adapter_path, &request, 0)
                .await?;
            return Ok(Box::new(tokio_stream::iter(events)));
        }

        // Multi-candidate: run sequentially (MLX uses all GPU memory)
        let mut all_events = Vec::new();
        for i in 0..n {
            match self
                .generate_single(mlx_model, adapter_path, &request, i as u16)
                .await
            {
                Ok(events) => all_events.extend(events),
                Err(e) => {
                    all_events.push(GenerationEvent::Error {
                        candidate_index: i as u16,
                        message: e.to_string(),
                    });
                }
            }
        }

        Ok(Box::new(tokio_stream::iter(all_events)))
    }
}
