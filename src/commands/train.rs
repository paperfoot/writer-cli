use serde::Serialize;

use writer_cli::backends::training::config::LoraConfig;
use writer_cli::backends::training::mlx_tune::{self, MlxTuneBackend};
use writer_cli::backends::training::TrainingBackend;
use writer_cli::backends::types::ModelId;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct TrainResult {
    profile: String,
    model: String,
    train_samples: usize,
    valid_samples: usize,
    steps: u32,
    final_loss: f32,
    adapter_path: String,
}

pub async fn run(ctx: Ctx, profile: Option<String>) -> Result<(), AppError> {
    let cfg = config::load()?;
    let profile_name = profile.unwrap_or_else(|| cfg.active_profile.clone());

    let profile_dir = config::profiles_dir().join(&profile_name);
    let corpus_path = profile_dir.join("samples/corpus.jsonl");

    if !corpus_path.exists() {
        return Err(AppError::Config(format!(
            "No corpus found for profile '{}'. Run: writer learn <files>",
            profile_name
        )));
    }

    // Prepare training data
    let train_data_dir = profile_dir.join("training_data");
    let (n_train, n_valid) =
        mlx_tune::prepare_training_data(&corpus_path, &train_data_dir, 0.1, cfg.training.dataset_format)
            .map_err(|e| AppError::Transient(e.to_string()))?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} Prepared {} train + {} validation samples",
            ">".blue(),
            n_train,
            n_valid
        );
    }

    // Parse model ID
    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model: {e}")))?;

    let adapter_path = profile_dir.join("adapters");

    let lora_config = LoraConfig {
        profile: profile_name.clone(),
        base_model: model_id.clone(),
        dataset_dir: train_data_dir,
        adapter_out: adapter_path.clone(),
        rank: cfg.training.rank,
        alpha: cfg.training.alpha,
        learning_rate: cfg.training.learning_rate,
        batch_size: cfg.training.batch_size,
        max_steps: cfg.training.max_steps,
        max_seq_len: cfg.training.max_seq_len,
        mask_prompt: cfg.training.mask_prompt,
    };

    // Create backend
    let backend = MlxTuneBackend::new().map_err(|e| AppError::Transient(e.to_string()))?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} Training LoRA adapter on {} ({} steps)...",
            ">".blue(),
            model_id,
            cfg.training.max_steps
        );
    }

    // Train with progress reporting
    let progress_fn = |p: writer_cli::backends::training::config::TrainingProgress| {
        eprint!(
            "\r  Step {}/{} | loss: {:.4} | lr: {:.6} | {:.1} it/s",
            p.step, p.total_steps, p.loss, p.learning_rate, p.tokens_per_second
        );
    };

    let artefact = backend
        .train_lora(lora_config, &progress_fn)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    if !ctx.format.is_json() {
        eprintln!(); // newline after progress
    }

    let result = TrainResult {
        profile: profile_name,
        model: model_id.to_string(),
        train_samples: n_train,
        valid_samples: n_valid,
        steps: artefact.steps,
        final_loss: artefact.final_loss,
        adapter_path: adapter_path.display().to_string(),
    };

    output::print_success_or(ctx, &result, |r| {
        use owo_colors::OwoColorize;
        println!(
            "{} Training complete! {} steps, final loss: {:.4}",
            "+".green(),
            r.steps,
            r.final_loss
        );
        println!("  adapter: {}", r.adapter_path.dimmed());
        println!();
        println!("Next: {}", "writer write \"your prompt\"".bold());
    });

    Ok(())
}
