//! `writer generate-pairs` — generate preference pairs for SimPO/DPO training.
//!
//! For each prompt in the suite, generates N candidates via the SFT model,
//! scores them against the stylometric fingerprint, and pairs best vs median.
//! Pairs below the margin threshold are discarded.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use writer_cli::backends::inference::mlx_worker::{MlxWorker, WorkerRequest};
use writer_cli::backends::types::ModelId;
use writer_cli::decoding::{logit_bias, prompts};
use writer_cli::stylometry::fingerprint::StylometricFingerprint;
use writer_cli::stylometry::{relevance, scoring};

use crate::config;
use crate::error::AppError;
use crate::output::Ctx;

#[derive(Debug, Serialize, Deserialize)]
struct PromptEntry {
    prompt: String,
    #[serde(default)]
    category: String,
}

#[derive(Debug, Serialize)]
struct PreferencePair {
    prompt: String,
    chosen: String,
    rejected: String,
    chosen_distance: f64,
    rejected_distance: f64,
    margin: f64,
    chosen_relevance: f64,
    seed: u64,
}

pub async fn run(
    ctx: Ctx,
    suite_path: PathBuf,
    n_candidates: u16,
    seeds: u16,
    min_margin: f64,
    output_path: PathBuf,
) -> Result<(), AppError> {
    let cfg = config::load()?;

    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model: {e}")))?;

    let profile_dir = config::profiles_dir().join(&cfg.active_profile);

    // Load fingerprint
    let fp_path = profile_dir.join("fingerprint.json");
    let fingerprint: StylometricFingerprint = {
        let fp_json = std::fs::read_to_string(&fp_path)
            .map_err(|e| AppError::Config(format!("Cannot read fingerprint: {e}")))?;
        serde_json::from_str(&fp_json)
            .map_err(|e| AppError::Config(format!("Invalid fingerprint: {e}")))?
    };

    // Load prompt suite
    let suite_text = std::fs::read_to_string(&suite_path)
        .map_err(|e| AppError::Config(format!("Cannot read prompt suite: {e}")))?;
    let prompts_list: Vec<PromptEntry> = serde_yaml::from_str(&suite_text)
        .map_err(|e| AppError::Config(format!("Invalid prompt suite YAML: {e}")))?;

    // Detect adapter
    let adapter_path = {
        let canonical = profile_dir.join("adapters");
        if canonical.join("adapters.safetensors").exists() {
            Some(canonical)
        } else {
            None
        }
    };

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} Generating preference pairs: {} prompts x {} seeds x {} candidates",
            ">".blue(),
            prompts_list.len(),
            seeds,
            n_candidates
        );
    }

    // Spawn persistent MLX worker
    let mut worker = MlxWorker::spawn(&model_id, adapter_path.as_deref())
        .await
        .map_err(|e| AppError::Transient(format!("Failed to spawn MLX worker: {e}")))?;

    let system = prompts::system_prompt(&fingerprint);
    let bias = logit_bias::from_fingerprint(&fingerprint, &cfg.decoding);

    let mut all_pairs: Vec<PreferencePair> = Vec::new();
    let total_gens = prompts_list.len() * seeds as usize * n_candidates as usize;
    let mut completed = 0usize;
    let start = std::time::Instant::now();

    for entry in &prompts_list {
        for seed_idx in 0..seeds {
            let rng_seed = 42u64 + seed_idx as u64 * 1000;

            // Generate N candidates
            let mut candidates: Vec<(String, f64, f64)> = Vec::new(); // (text, distance, relevance)

            for cand_idx in 0..n_candidates {
                let candidate_seed = rng_seed + cand_idx as u64;
                let req = WorkerRequest {
                    prompt: entry.prompt.clone(),
                    system_prompt: Some(system.clone()),
                    prompt_mode: "chat".to_string(),
                    max_tokens: cfg.decoding.max_tokens,
                    temperature: cfg.inference.temperature,
                    top_p: 0.92,
                    repetition_penalty: cfg.decoding.repetition_penalty,
                    seed: Some(candidate_seed),
                    logit_bias: bias.clone(),
                };

                match worker.generate(&req).await {
                    Ok(events) => {
                        let text = events
                            .iter()
                            .filter_map(|e| match e {
                                writer_cli::backends::inference::response::GenerationEvent::Done {
                                    full_text,
                                    ..
                                } => Some(full_text.clone()),
                                _ => None,
                            })
                            .next()
                            .unwrap_or_default();

                        if !text.is_empty() {
                            let report = scoring::distance(&text, &fingerprint);
                            let rel = relevance::score(&entry.prompt, &text);
                            candidates.push((text, report.overall, rel));
                        }
                    }
                    Err(e) => {
                        eprintln!("  Warning: generation failed: {e}");
                    }
                }

                completed += 1;
                let elapsed = start.elapsed().as_secs_f64();
                let eta = if completed > 0 {
                    (total_gens - completed) as f64 * elapsed / completed as f64
                } else {
                    0.0
                };
                eprint!(
                    "\r  [{}/{}] ETA: {:.0}m {:.0}s",
                    completed,
                    total_gens,
                    eta / 60.0,
                    eta % 60.0
                );
            }

            if candidates.len() < 2 {
                continue;
            }

            // Sort by distance (lower = better style match)
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let (chosen_text, chosen_dist, chosen_rel) = &candidates[0];
            let median_idx = candidates.len() / 2;
            let (rejected_text, rejected_dist, _) = &candidates[median_idx];

            let margin = rejected_dist - chosen_dist;

            // Filter by margin and relevance
            if margin >= min_margin && *chosen_rel >= 0.6 {
                all_pairs.push(PreferencePair {
                    prompt: entry.prompt.clone(),
                    chosen: chosen_text.clone(),
                    rejected: rejected_text.clone(),
                    chosen_distance: *chosen_dist,
                    rejected_distance: *rejected_dist,
                    margin,
                    chosen_relevance: *chosen_rel,
                    seed: rng_seed,
                });
            }
        }
    }

    eprintln!();

    // Write output
    let mut output_lines = String::new();
    for pair in &all_pairs {
        if let Ok(line) = serde_json::to_string(pair) {
            output_lines.push_str(&line);
            output_lines.push('\n');
        }
    }
    std::fs::write(&output_path, &output_lines)?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} Generated {} preference pairs (saved to {})",
            ">".green(),
            all_pairs.len(),
            output_path.display()
        );
        if !all_pairs.is_empty() {
            let avg_margin: f64 =
                all_pairs.iter().map(|p| p.margin).sum::<f64>() / all_pairs.len() as f64;
            println!(
                "  Average margin: {:.3}, min: {:.3}, max: {:.3}",
                avg_margin,
                all_pairs.iter().map(|p| p.margin).fold(f64::MAX, f64::min),
                all_pairs.iter().map(|p| p.margin).fold(0.0, f64::max),
            );
        }
    }

    Ok(())
}
