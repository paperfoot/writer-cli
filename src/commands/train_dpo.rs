//! `writer train-dpo` — SimPO/DPO preference training on preference pairs.
use std::path::PathBuf;

use writer_cli::backends::training::config::{DpoConfig, PreferenceMethod};
use writer_cli::backends::training::mlx_tune::MlxTuneBackend;
use writer_cli::backends::training::TrainingBackend;
use writer_cli::backends::types::ModelId;

use crate::config;
use crate::error::AppError;
use crate::output::Ctx;

pub async fn run(
    ctx: Ctx,
    pairs_path: PathBuf,
    method: String,
    lr: f32,
    steps: u32,
    gamma: f32,
    beta: f32,
) -> Result<(), AppError> {
    let cfg = config::load()?;

    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model: {e}")))?;

    let profile_dir = config::profiles_dir().join(&cfg.active_profile);

    // Check SFT adapter exists
    let sft_adapter_path = profile_dir.join("adapters");
    let base_adapter = if sft_adapter_path.join("adapters.safetensors").exists() {
        Some(sft_adapter_path.clone())
    } else {
        None
    };

    // Output to a separate directory so we don't overwrite SFT adapter
    let dpo_adapter_out = profile_dir.join("adapters-dpo");

    let method_enum = match method.to_lowercase().as_str() {
        "simpo" => PreferenceMethod::Simpo,
        "dpo" => PreferenceMethod::Dpo,
        other => {
            return Err(AppError::Config(format!(
                "Unknown preference method: {other}. Use 'simpo' or 'dpo'."
            )));
        }
    };

    let dpo_config = DpoConfig {
        profile: cfg.active_profile.clone(),
        base_model: model_id.clone(),
        preference_dataset: pairs_path.clone(),
        adapter_out: dpo_adapter_out.clone(),
        base_adapter,
        method: method_enum,
        beta,
        gamma,
        learning_rate: lr,
        batch_size: 1,
        max_steps: steps,
        max_seq_len: cfg.training.max_seq_len,
    };

    let backend = MlxTuneBackend::new().map_err(|e| AppError::Transient(e.to_string()))?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} Training {} adapter ({} steps, lr={})...",
            ">".blue(),
            method.to_uppercase(),
            steps,
            lr
        );
        println!(
            "  Base adapter: {}",
            dpo_config
                .base_adapter
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "none".to_string())
        );
        println!("  Pairs: {}", pairs_path.display());
    }

    let progress_fn =
        |p: writer_cli::backends::training::config::TrainingProgress| {
            let pct = if p.total_steps > 0 {
                p.step as f32 / p.total_steps as f32 * 100.0
            } else {
                0.0
            };
            eprint!(
                "\r  step {}/{} ({:.0}%) | loss: {:.4}",
                p.step, p.total_steps, pct, p.loss
            );
        };

    let artifact = backend
        .train_dpo(dpo_config, &progress_fn)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    eprintln!();

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} {} adapter saved to {} (steps: {}, loss: {:.4})",
            ">".green(),
            method.to_uppercase(),
            dpo_adapter_out.display(),
            artifact.steps,
            artifact.final_loss
        );
        println!(
            "  To use: copy to {} or update config",
            profile_dir.join("adapters").display()
        );
    }

    Ok(())
}
