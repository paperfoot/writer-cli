use serde::Serialize;

use writer_cli::backends::inference::mlx::MlxBackend;
use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::types::{AdapterRef, ModelId};
use writer_cli::decoding;
use writer_cli::decoding::prompts;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct WriteResult {
    text: String,
}

#[derive(Serialize)]
struct WriteResultVerbose {
    text: String,
    distance: f64,
    model: String,
    adapter: bool,
    backend: String,
    tokens: u32,
    elapsed_ms: u64,
    candidates: usize,
}

/// Detect the canonical adapter for the active profile.
/// Only uses the `adapters/` directory — adapter must be trained for
/// the currently configured base model. Legacy `adapters-*/` checkpoint
/// directories are ignored to avoid model/adapter mismatches.
fn detect_adapter(profile_dir: &std::path::Path) -> Option<AdapterRef> {
    let canonical = profile_dir.join("adapters");
    if canonical.join("adapters.safetensors").exists() {
        return Some(AdapterRef::new(
            profile_dir
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            canonical,
        ));
    }
    None
}

pub async fn run(
    ctx: Ctx,
    prompt: String,
    max_tokens_override: Option<u32>,
    candidates_override: Option<u16>,
    verbose: bool,
    raw: bool,
) -> Result<(), AppError> {
    let mut cfg = config::load()?;

    // Apply CLI overrides
    if let Some(mt) = max_tokens_override {
        cfg.decoding.max_tokens = mt;
    }
    if let Some(n) = candidates_override {
        cfg.decoding.n_candidates = n;
    }

    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model in config: {e}")))?;

    let profile_dir = config::profiles_dir().join(&cfg.active_profile);
    let adapter = detect_adapter(&profile_dir);

    // Choose backend: MLX when adapter present (safetensors), Ollama otherwise
    let (backend_box, backend_name): (Box<dyn InferenceBackend>, &str) = if adapter.is_some() {
        match MlxBackend::new() {
            Ok(mlx) => (Box::new(mlx), "mlx"),
            Err(e) => {
                eprintln!(
                    "Warning: adapter found but MLX backend unavailable ({e}), falling back to Ollama"
                );
                let ollama = OllamaBackend::new(&cfg.inference.ollama_url);
                ollama
                    .ping()
                    .await
                    .map_err(|e| AppError::Transient(e.to_string()))?;
                (Box::new(ollama), "ollama")
            }
        }
    } else {
        let ollama = OllamaBackend::new(&cfg.inference.ollama_url);
        ollama
            .ping()
            .await
            .map_err(|e| AppError::Transient(e.to_string()))?;
        (Box::new(ollama), "ollama")
    };

    let handle = backend_box
        .load_model(&model_id)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    // Load fingerprint if available
    let fp_path = profile_dir.join("fingerprint.json");
    let fingerprint = if fp_path.exists() {
        let fp_json = std::fs::read_to_string(&fp_path)?;
        serde_json::from_str::<StylometricFingerprint>(&fp_json)
            .map_err(|e| AppError::Config(format!("Invalid fingerprint.json: {e}")))?
    } else {
        StylometricFingerprint::default()
    };

    // In raw mode: no system prompt, no prompt wrapping, raw prompt_mode for MLX
    let system = if raw {
        None
    } else if fingerprint.word_count > 0 {
        Some(prompts::system_prompt(&fingerprint))
    } else {
        None
    };

    let write_prompt = if raw {
        prompt.clone()
    } else {
        prompts::write_prompt(&prompt)
    };

    let prompt_mode = if raw {
        Some("raw".to_string())
    } else {
        match cfg.inference.prompt_mode {
            writer_cli::config::PromptMode::Raw => Some("raw".to_string()),
            writer_cli::config::PromptMode::Chat => None,
        }
    };

    // Run through decoding pipeline (adapter is passed via the request)
    let result = decoding::run(
        backend_box.as_ref(),
        &handle,
        &model_id,
        &fingerprint,
        &cfg.decoding,
        &write_prompt,
        system.as_deref(),
        adapter.as_ref(),
        prompt_mode.as_deref(),
    )
    .await
    .map_err(|e| AppError::Transient(e.to_string()))?;

    if !ctx.format.is_json() {
        print!("{}", result.text);
        println!();
    } else if verbose {
        let output = WriteResultVerbose {
            text: result.text,
            distance: (result.distance * 1000.0).round() / 1000.0,
            model: model_id.to_string(),
            adapter: adapter.is_some(),
            backend: backend_name.to_string(),
            tokens: result.tokens_generated,
            elapsed_ms: result.elapsed_ms,
            candidates: result.candidates_generated,
        };
        output::print_success_or(ctx, &output, |_| {});
    } else {
        let output = WriteResult {
            text: result.text,
        };
        output::print_success_or(ctx, &output, |_| {});
    }

    Ok(())
}
