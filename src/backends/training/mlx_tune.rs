//! mlx-tune training backend — LoRA fine-tuning on Apple Silicon.
//!
//! Reference: mlx-lm (https://github.com/ml-explore/mlx-lm, 4.8k stars)
//! Uses mlx_lm.lora for LoRA/DoRA fine-tuning via subprocess.
use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use serde::Serialize;

use super::artefact::AdapterArtifact;
use super::config::{DpoConfig, LoraConfig, TrainingProgress};
use super::{TrainingBackend, TrainingError};
use crate::backends::types::AdapterRef;

pub struct MlxTuneBackend {
    mlx_lm_path: PathBuf,
}

impl MlxTuneBackend {
    pub fn new() -> Result<Self, TrainingError> {
        // Find mlx_lm.lora on PATH
        let output = std::process::Command::new("which")
            .arg("mlx_lm.lora")
            .output()
            .map_err(|_| {
                TrainingError::Unavailable(
                    "mlx_lm.lora not found. Install: pip install mlx-lm".to_string(),
                )
            })?;

        if !output.status.success() {
            return Err(TrainingError::Unavailable(
                "mlx_lm.lora not found on PATH. Install: pip install mlx-lm".to_string(),
            ));
        }

        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(Self {
            mlx_lm_path: PathBuf::from(path),
        })
    }
}

#[async_trait]
impl TrainingBackend for MlxTuneBackend {
    fn name(&self) -> &str {
        "mlx-tune"
    }

    async fn train_lora(
        &self,
        config: LoraConfig,
        on_progress: &(dyn Fn(TrainingProgress) + Sync),
    ) -> Result<AdapterArtifact, TrainingError> {
        // Create temp dir for training data
        let data_dir = config.dataset_dir.clone();
        std::fs::create_dir_all(&data_dir)?;

        // Build adapter output path
        std::fs::create_dir_all(config.adapter_out.parent().unwrap_or(Path::new(".")))?;

        // Resolve to the quantized MLX community model for Apple Silicon training.
        // google/gemma-4-26b -> mlx-community/gemma-4-26b-a4b-it-4bit
        let model_name = resolve_mlx_model(&config.base_model);

        // Write LoRA config YAML so rank/alpha are passed through.
        // mlx_lm.lora defaults to rank=8, scale=20.0 — our config may differ.
        // scale = alpha / rank (mlx-lm convention).
        let lora_scale = if config.rank > 0 {
            config.alpha / config.rank as f32
        } else {
            20.0
        };
        let lora_config_yaml = format!(
            "lora_parameters:\n  rank: {}\n  scale: {:.1}\n  dropout: 0.0\n",
            config.rank, lora_scale
        );
        let lora_config_path = config
            .adapter_out
            .parent()
            .unwrap_or(Path::new("."))
            .join("lora_config.yaml");
        std::fs::write(&lora_config_path, &lora_config_yaml)?;

        // Run mlx_lm.lora
        let mut cmd = tokio::process::Command::new(&self.mlx_lm_path);
        cmd.arg("--train")
            .arg("--model")
            .arg(&model_name)
            .arg("--data")
            .arg(&data_dir)
            .arg("--adapter-path")
            .arg(&config.adapter_out)
            .arg("--iters")
            .arg(config.max_steps.to_string())
            .arg("--batch-size")
            .arg(config.batch_size.to_string())
            .arg("--learning-rate")
            .arg(config.learning_rate.to_string())
            .arg("--max-seq-length")
            .arg(config.max_seq_len.to_string())
            .arg("-c")
            .arg(&lora_config_path);

        if config.mask_prompt {
            cmd.arg("--mask-prompt");
        }

        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                TrainingError::TrainingFailed(format!("Failed to spawn mlx_lm.lora: {e}"))
            })?;

        // Parse progress from stdout
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Read stderr in background for error capture
        let stderr_handle = stderr.map(|stderr| {
            tokio::spawn(async move {
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = Vec::new();
                use tokio::io::AsyncBufReadExt;
                let mut line_reader = reader.lines();
                while let Ok(Some(line)) = line_reader.next_line().await {
                    lines.push(line);
                }
                lines
            })
        });

        // Parse stdout progress lines
        let mut last_loss = 0.0f32;
        let mut last_step = 0u32;
        if let Some(stdout) = stdout {
            let mut reader = tokio::io::BufReader::new(stdout);
            use tokio::io::AsyncBufReadExt;
            let mut line_buf = String::new();
            loop {
                line_buf.clear();
                match reader.read_line(&mut line_buf).await {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        if let Some(progress) = parse_progress_line(&line_buf, config.max_steps) {
                            last_loss = progress.loss;
                            last_step = progress.step;
                            on_progress(progress);
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        let status = child
            .wait()
            .await
            .map_err(|e| TrainingError::TrainingFailed(format!("Process wait failed: {e}")))?;

        if !status.success() {
            let stderr_lines = if let Some(handle) = stderr_handle {
                handle.await.unwrap_or_default()
            } else {
                Vec::new()
            };
            return Err(TrainingError::TrainingFailed(format!(
                "mlx_lm.lora exited with {}: {}",
                status,
                stderr_lines.join("\n")
            )));
        }

        Ok(AdapterArtifact {
            adapter: AdapterRef::new(config.profile, config.adapter_out),
            base_model: config.base_model,
            steps: last_step,
            final_loss: last_loss,
            training_seconds: 0, // TODO: track
        })
    }

    async fn train_dpo(
        &self,
        config: DpoConfig,
        on_progress: &(dyn Fn(TrainingProgress) + Sync),
    ) -> Result<AdapterArtifact, TrainingError> {
        std::fs::create_dir_all(config.adapter_out.parent().unwrap_or(Path::new(".")))?;

        let model_name = resolve_mlx_model(&config.base_model);
        let lora_scale = 2.0f32; // alpha / rank, matching SFT default

        let dpo_config = serde_json::json!({
            "model": model_name,
            "dataset_path": config.preference_dataset,
            "adapter_out": config.adapter_out,
            "resume_adapter": config.base_adapter,
            "method": match config.method {
                super::config::PreferenceMethod::Simpo => "simpo",
                super::config::PreferenceMethod::Dpo => "dpo",
            },
            "beta": config.beta,
            "gamma": config.gamma,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "max_seq_len": config.max_seq_len,
            "lora_rank": 16,
            "lora_scale": lora_scale,
        });

        let script = find_dpo_script()?;
        let config_json = serde_json::to_string(&dpo_config)
            .map_err(|e| TrainingError::TrainingFailed(format!("Serialize DPO config: {e}")))?;

        let mut child = tokio::process::Command::new("python3")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                TrainingError::TrainingFailed(format!("Failed to spawn mlx_dpo.py: {e}"))
            })?;

        // Write config to stdin
        if let Some(mut stdin) = child.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(config_json.as_bytes()).await.map_err(|e| {
                TrainingError::TrainingFailed(format!("Failed to write DPO config: {e}"))
            })?;
            drop(stdin);
        }

        // Read stderr in background
        let stderr = child.stderr.take();
        let stderr_handle = stderr.map(|stderr| {
            tokio::spawn(async move {
                let reader = tokio::io::BufReader::new(stderr);
                let mut lines = Vec::new();
                use tokio::io::AsyncBufReadExt;
                let mut line_reader = reader.lines();
                while let Ok(Some(line)) = line_reader.next_line().await {
                    eprintln!("  [dpo] {line}");
                    lines.push(line);
                }
                lines
            })
        });

        // Parse progress from stdout
        let mut last_loss = 0.0f32;
        let mut last_step = 0u32;
        if let Some(stdout) = child.stdout.take() {
            let mut reader = tokio::io::BufReader::new(stdout);
            use tokio::io::AsyncBufReadExt;
            let mut line_buf = String::new();
            loop {
                line_buf.clear();
                match reader.read_line(&mut line_buf).await {
                    Ok(0) => break,
                    Ok(_) => {
                        if let Ok(progress) =
                            serde_json::from_str::<serde_json::Value>(line_buf.trim())
                        {
                            let step = progress["step"].as_u64().unwrap_or(0) as u32;
                            let total = progress["total_steps"].as_u64().unwrap_or(0) as u32;
                            let loss = progress["loss"].as_f64().unwrap_or(0.0) as f32;
                            let lr = progress["learning_rate"].as_f64().unwrap_or(0.0) as f32;
                            last_loss = loss;
                            last_step = step;
                            on_progress(TrainingProgress {
                                step,
                                total_steps: total,
                                loss,
                                learning_rate: lr,
                                tokens_per_second: 0.0,
                            });
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        let status = child
            .wait()
            .await
            .map_err(|e| TrainingError::TrainingFailed(format!("Process wait failed: {e}")))?;

        if !status.success() {
            let stderr_lines = if let Some(handle) = stderr_handle {
                handle.await.unwrap_or_default()
            } else {
                Vec::new()
            };
            return Err(TrainingError::TrainingFailed(format!(
                "mlx_dpo.py exited with {}: {}",
                status,
                stderr_lines.join("\n")
            )));
        }

        Ok(AdapterArtifact {
            adapter: AdapterRef::new(config.profile, config.adapter_out),
            base_model: config.base_model,
            steps: last_step,
            final_loss: last_loss,
            training_seconds: 0,
        })
    }
}

fn parse_progress_line(line: &str, total_steps: u32) -> Option<TrainingProgress> {
    // Parse: "Iter 100: Train loss 2.345, Learning Rate 0.0001, It/sec 5.678, ..."
    if !line.starts_with("Iter ") {
        return None;
    }

    let step = line
        .split(':')
        .next()?
        .trim_start_matches("Iter ")
        .trim()
        .parse::<u32>()
        .ok()?;

    let loss = line
        .split("Train loss ")
        .nth(1)?
        .split(',')
        .next()?
        .trim()
        .parse::<f32>()
        .ok()?;

    let lr = line
        .split("Learning Rate ")
        .nth(1)?
        .split(',')
        .next()?
        .trim()
        .parse::<f32>()
        .unwrap_or(0.0);

    let tps = line
        .split("It/sec ")
        .nth(1)?
        .split(',')
        .next()?
        .trim()
        .parse::<f32>()
        .unwrap_or(0.0);

    Some(TrainingProgress {
        step,
        total_steps,
        loss,
        learning_rate: lr,
        tokens_per_second: tps,
    })
}

/// Prepare training data in the configured dataset format.
///
/// Formats (per mlx-lm LORA.md):
/// - chat: {"messages": [{role, content}, ...]} — uses chat template
/// - completions: {"prompt": "...", "completion": "..."} — supports mask_prompt
/// - text: {"text": "..."} — raw continuation, fully custom formatting
pub fn prepare_training_data(
    corpus_jsonl: &Path,
    output_dir: &Path,
    holdout_ratio: f64,
    format: crate::config::DatasetFormat,
) -> Result<(usize, usize), TrainingError> {
    use crate::corpus::sample::Sample;

    let content = std::fs::read_to_string(corpus_jsonl)
        .map_err(|e| TrainingError::Dataset(format!("Cannot read corpus: {e}")))?;

    let samples: Vec<Sample> = content
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .filter(|s: &Sample| s.word_count() >= 20) // Skip tiny samples
        .collect();

    if samples.is_empty() {
        return Err(TrainingError::Dataset(
            "No samples found in corpus".to_string(),
        ));
    }

    let n_holdout = (samples.len() as f64 * holdout_ratio).ceil() as usize;
    let n_holdout = n_holdout.max(1).min(samples.len() / 2);

    let (valid_samples, train_samples) = samples.split_at(n_holdout);

    std::fs::create_dir_all(output_dir)?;

    let writer_fn: fn(&Path, &[Sample]) -> Result<(), TrainingError> = match format {
        crate::config::DatasetFormat::Chat => write_chat_jsonl,
        crate::config::DatasetFormat::Completions => write_completion_jsonl,
        crate::config::DatasetFormat::Text => write_text_jsonl,
    };

    writer_fn(&output_dir.join("train.jsonl"), train_samples)?;
    writer_fn(&output_dir.join("valid.jsonl"), valid_samples)?;
    writer_fn(&output_dir.join("test.jsonl"), valid_samples)?;

    Ok((train_samples.len(), valid_samples.len()))
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatExample {
    messages: Vec<ChatMessage>,
}

fn find_dpo_script() -> Result<PathBuf, TrainingError> {
    if let Ok(p) = std::env::var("WRITER_DPO_SCRIPT") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        let candidates = [
            exe.parent()
                .unwrap_or(Path::new("."))
                .join("../scripts/mlx_dpo.py"),
            exe.parent()
                .unwrap_or(Path::new("."))
                .join("../../scripts/mlx_dpo.py"),
        ];
        for c in &candidates {
            if let Ok(canonical) = c.canonicalize() {
                if canonical.exists() {
                    return Ok(canonical);
                }
            }
        }
    }

    let dev_path = PathBuf::from("scripts/mlx_dpo.py");
    if dev_path.exists() {
        return Ok(dev_path.canonicalize().unwrap_or(dev_path));
    }

    Err(TrainingError::TrainingFailed(
        "scripts/mlx_dpo.py not found. Set WRITER_DPO_SCRIPT env var.".into(),
    ))
}

/// Map a writer ModelId to the quantized MLX community model path.
/// mlx-lm needs the HuggingFace repo path for the quantized weights.
fn resolve_mlx_model(model_id: &crate::backends::types::ModelId) -> String {
    let owner = model_id.owner();
    let name = model_id.name();

    // Already an MLX community model — pass through
    if owner == "mlx-community" {
        return format!("{owner}/{name}");
    }

    // Known mappings for common models
    if name.contains("gemma-4-26b") || name.contains("gemma4-26b") {
        return "mlx-community/gemma-4-26b-a4b-it-4bit".to_string();
    }
    if name.contains("gemma-3-4b") || name.contains("gemma3-4b") {
        return "mlx-community/gemma-3-4b-it-4bit".to_string();
    }

    // Default: try mlx-community with -4bit suffix
    format!("mlx-community/{name}-4bit")
}

fn write_chat_jsonl(
    path: &Path,
    samples: &[crate::corpus::sample::Sample],
) -> Result<(), TrainingError> {
    let mut output = String::new();
    for sample in samples {
        let context = sample.metadata.context_tag.as_deref().unwrap_or("longform");

        let example = ChatExample {
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: format!("Write a {context} passage in your natural voice."),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: sample.content.clone(),
                },
            ],
        };

        if let Ok(line) = serde_json::to_string(&example) {
            output.push_str(&line);
            output.push('\n');
        }
    }
    std::fs::write(path, output).map_err(TrainingError::Io)
}

/// Completions format: {"prompt": "...", "completion": "..."}
/// Supports mask_prompt in mlx-lm to train only on the completion portion.
fn write_completion_jsonl(
    path: &Path,
    samples: &[crate::corpus::sample::Sample],
) -> Result<(), TrainingError> {
    let mut output = String::new();
    for sample in samples {
        let context = sample.metadata.context_tag.as_deref().unwrap_or("longform");

        let example = CompletionExample {
            prompt: format!("Write a {context} passage in your natural voice.\n\n"),
            completion: sample.content.clone(),
        };

        if let Ok(line) = serde_json::to_string(&example) {
            output.push_str(&line);
            output.push('\n');
        }
    }
    std::fs::write(path, output).map_err(TrainingError::Io)
}

/// Text format: {"text": "..."} — raw continuation, no chat wrapping.
/// mlx-lm treats this as pure language modeling with no template applied.
fn write_text_jsonl(
    path: &Path,
    samples: &[crate::corpus::sample::Sample],
) -> Result<(), TrainingError> {
    let mut output = String::new();
    for sample in samples {
        let example = TextExample {
            text: sample.content.clone(),
        };

        if let Ok(line) = serde_json::to_string(&example) {
            output.push_str(&line);
            output.push('\n');
        }
    }
    std::fs::write(path, output).map_err(TrainingError::Io)
}

#[derive(Serialize)]
struct CompletionExample {
    prompt: String,
    completion: String,
}

#[derive(Serialize)]
struct TextExample {
    text: String,
}
