use serde::Serialize;

use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::types::ModelId;
use writer_cli::decoding;
use writer_cli::decoding::prompts;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct WriteResult {
    text: String,
    model: String,
    tokens_generated: u32,
    elapsed_ms: u64,
    stylometric_distance: f64,
    candidates_generated: usize,
    regenerations: usize,
}

pub async fn run(ctx: Ctx, prompt: String) -> Result<(), AppError> {
    let cfg = config::load()?;
    let backend = OllamaBackend::new(&cfg.inference.ollama_url);

    backend
        .ping()
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model in config: {e}")))?;

    let handle = backend
        .load_model(&model_id)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    // Load fingerprint if available
    let profile_dir = config::profiles_dir().join(&cfg.active_profile);
    let fp_path = profile_dir.join("fingerprint.json");
    let fingerprint = if fp_path.exists() {
        let fp_json = std::fs::read_to_string(&fp_path)?;
        serde_json::from_str::<StylometricFingerprint>(&fp_json)
            .map_err(|e| AppError::Config(format!("Invalid fingerprint.json: {e}")))?
    } else {
        StylometricFingerprint::default()
    };

    // Build system prompt with stylometric priming
    let system = if fingerprint.word_count > 0 {
        Some(prompts::system_prompt(&fingerprint))
    } else {
        None
    };

    let write_prompt = prompts::write_prompt(&prompt);

    // Run through decoding pipeline
    let result = decoding::run(
        &backend,
        &handle,
        &model_id,
        &fingerprint,
        &cfg.decoding,
        &write_prompt,
        system.as_deref(),
    )
    .await
    .map_err(|e| AppError::Transient(e.to_string()))?;

    if !ctx.format.is_json() {
        print!("{}", result.text);
        println!();
    }

    let output_result = WriteResult {
        text: result.text,
        model: model_id.to_string(),
        tokens_generated: result.tokens_generated,
        elapsed_ms: result.elapsed_ms,
        stylometric_distance: result.distance,
        candidates_generated: result.candidates_generated,
        regenerations: result.regenerations,
    };

    if ctx.format.is_json() {
        output::print_success_or(ctx, &output_result, |_| {});
    }

    Ok(())
}
