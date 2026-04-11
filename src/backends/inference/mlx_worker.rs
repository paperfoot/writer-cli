//! Persistent MLX worker — keeps model loaded between generations.
//!
//! Used by ablation and eval to avoid reloading the 15GB model for each request.
//! Protocol: line-delimited JSON over stdin/stdout with the mlx_worker.py script.
use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};

use super::response::{FinishReason, GenerationEvent, UsageStats};
use crate::backends::types::ModelId;

pub struct MlxWorker {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    #[allow(dead_code)]
    child: Child,
}

#[derive(Debug, Serialize)]
struct WorkerStartup {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WorkerRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub prompt_mode: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub logit_bias: std::collections::HashMap<String, f32>,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    text: Option<String>,
    error: Option<String>,
    #[allow(dead_code)]
    status: Option<String>,
    prompt_tokens: Option<u32>,
    generation_tokens: Option<u32>,
    #[allow(dead_code)]
    generation_tps: Option<f64>,
    #[allow(dead_code)]
    peak_memory_gb: Option<f64>,
    finish_reason: Option<String>,
    elapsed_ms: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("worker unavailable: {0}")]
    Unavailable(String),
    #[error("worker error: {0}")]
    WorkerFailed(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl MlxWorker {
    /// Spawn a persistent worker for the given model and optional adapter.
    /// The worker loads the model once and serves requests until dropped.
    pub async fn spawn(
        model_id: &ModelId,
        adapter_path: Option<&Path>,
    ) -> Result<Self, WorkerError> {
        let script = find_worker_script()?;
        let mlx_model = resolve_mlx_model(model_id);

        let mut child = tokio::process::Command::new("python3")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| WorkerError::Unavailable(format!("Failed to spawn mlx_worker.py: {e}")))?;

        let mut stdin = child.stdin.take().ok_or_else(|| {
            WorkerError::Unavailable("Failed to open worker stdin".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            WorkerError::Unavailable("Failed to open worker stdout".to_string())
        })?;

        // Send startup config
        let startup = WorkerStartup {
            model: mlx_model,
            adapter_path: adapter_path.map(|p| p.to_string_lossy().to_string()),
        };
        let startup_json = serde_json::to_string(&startup)
            .map_err(|e| WorkerError::WorkerFailed(format!("Serialize startup: {e}")))?;
        stdin.write_all(startup_json.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.flush().await?;

        let mut reader = BufReader::new(stdout);

        // Wait for ready signal
        let mut ready_line = String::new();
        reader.read_line(&mut ready_line).await?;

        let ready: serde_json::Value = serde_json::from_str(ready_line.trim())
            .map_err(|e| WorkerError::WorkerFailed(format!("Bad ready signal: {e}: {ready_line}")))?;

        if ready.get("status").and_then(|s| s.as_str()) != Some("ready") {
            return Err(WorkerError::WorkerFailed(format!(
                "Unexpected startup response: {ready_line}"
            )));
        }

        Ok(Self {
            stdin,
            stdout: reader,
            child,
        })
    }

    /// Send a generation request and wait for the response.
    pub async fn generate(&mut self, request: &WorkerRequest) -> Result<Vec<GenerationEvent>, WorkerError> {
        let req_json = serde_json::to_string(request)
            .map_err(|e| WorkerError::WorkerFailed(format!("Serialize request: {e}")))?;

        self.stdin.write_all(req_json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        let mut resp_line = String::new();
        self.stdout.read_line(&mut resp_line).await?;

        if resp_line.trim().is_empty() {
            return Err(WorkerError::WorkerFailed("Empty response from worker".to_string()));
        }

        let resp: WorkerResponse = serde_json::from_str(resp_line.trim())
            .map_err(|e| WorkerError::WorkerFailed(format!(
                "Parse response: {e}\nRaw: {resp_line}"
            )))?;

        if let Some(err) = resp.error {
            return Err(WorkerError::WorkerFailed(err));
        }

        let text = resp.text.unwrap_or_default();
        let finish = match resp.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::MaxTokens,
            _ => FinishReason::Stop,
        };

        Ok(vec![
            GenerationEvent::Token {
                candidate_index: 0,
                text: text.clone(),
                logprob: 0.0,
            },
            GenerationEvent::Done {
                candidate_index: 0,
                finish_reason: finish,
                usage: UsageStats {
                    prompt_tokens: resp.prompt_tokens.unwrap_or(0),
                    generated_tokens: resp.generation_tokens.unwrap_or(0),
                    elapsed_ms: resp.elapsed_ms.unwrap_or(0),
                },
                full_text: text,
            },
        ])
    }
}

fn find_worker_script() -> Result<PathBuf, WorkerError> {
    if let Ok(p) = std::env::var("WRITER_MLX_WORKER") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        let candidates = [
            exe.parent().unwrap_or(Path::new(".")).join("../scripts/mlx_worker.py"),
            exe.parent().unwrap_or(Path::new(".")).join("../../scripts/mlx_worker.py"),
        ];
        for c in &candidates {
            if let Ok(canonical) = c.canonicalize() {
                if canonical.exists() {
                    return Ok(canonical);
                }
            }
        }
    }

    let dev_path = PathBuf::from("scripts/mlx_worker.py");
    if dev_path.exists() {
        return Ok(dev_path.canonicalize().unwrap_or(dev_path));
    }

    Err(WorkerError::Unavailable(
        "scripts/mlx_worker.py not found. Set WRITER_MLX_WORKER env var.".into(),
    ))
}

fn resolve_mlx_model(model_id: &ModelId) -> String {
    let name = model_id.name();
    if model_id.owner() == "mlx-community" {
        return format!("{}/{}", model_id.owner(), name);
    }
    if name.contains("gemma-4-26b") || name.contains("gemma4-26b") {
        return "mlx-community/gemma-4-26b-a4b-it-4bit".to_string();
    }
    if name.contains("gemma-3-4b") || name.contains("gemma3-4b") {
        return "mlx-community/gemma-3-4b-it-4bit".to_string();
    }
    format!("mlx-community/{}-4bit", name)
}
