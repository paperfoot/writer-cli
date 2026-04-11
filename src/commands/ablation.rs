//! `writer ablation` — automated format ablation experiment.
//!
//! Trains 3 adapters (chat, completions+mask, text) on the same corpus split,
//! then evaluates each under multiple inference modes (wrapped, raw).
//! Produces a comparison report to guide format selection.
use std::path::{Path, PathBuf};

use serde::Serialize;

use writer_cli::backends::inference::mlx::MlxBackend;
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::training::config::LoraConfig;
use writer_cli::backends::training::mlx_tune::{self, MlxTuneBackend};
use writer_cli::backends::training::TrainingBackend;
use writer_cli::backends::types::ModelId;
use writer_cli::config::DatasetFormat;
use writer_cli::decoding;
use writer_cli::decoding::prompts;
use writer_cli::stylometry::features::punctuation::PunctuationStats;
use writer_cli::stylometry::features::readability::ReadabilityStats;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;

use crate::config;
use crate::error::AppError;
use crate::output::Ctx;

// ── Ablation configuration ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct AblationConfig {
    format: DatasetFormat,
    mask_prompt: bool,
    name: &'static str,
}

const ABLATION_CONFIGS: &[AblationConfig] = &[
    AblationConfig {
        format: DatasetFormat::Chat,
        mask_prompt: false,
        name: "chat",
    },
    AblationConfig {
        format: DatasetFormat::Completions,
        mask_prompt: true,
        name: "completions-masked",
    },
    AblationConfig {
        format: DatasetFormat::Text,
        mask_prompt: false,
        name: "text",
    },
];

#[derive(Debug, Clone, Copy)]
struct InferenceMode {
    raw: bool,
    name: &'static str,
}

const INFERENCE_MODES: &[InferenceMode] = &[
    InferenceMode {
        raw: false,
        name: "wrapped",
    },
    InferenceMode {
        raw: true,
        name: "raw",
    },
];

// ── Result types ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AblationResult {
    training_format: String,
    inference_mode: String,
    mean_style_distance: f64,
    median_style_distance: f64,
    mean_fk_grade: f64,
    mean_questions_per_1k: f64,
    mean_exclamations_per_1k: f64,
    mean_canon_leakage: f64,
    generations: usize,
    final_loss: f32,
}

#[derive(Debug, Serialize)]
struct AblationSummary {
    configs_tested: usize,
    inference_modes: usize,
    total_combinations: usize,
    best_combination: String,
    best_style_distance: f64,
    results: Vec<AblationResult>,
}

// ── Main entry point ──────────────────────────────────────────────────────

pub async fn run(
    ctx: Ctx,
    steps: u32,
    seeds: u16,
    suite_path: PathBuf,
    output_dir: PathBuf,
    eval_only: bool,
) -> Result<(), AppError> {
    let cfg = config::load()?;
    let profile_dir = config::profiles_dir().join(&cfg.active_profile);
    let corpus_path = profile_dir.join("samples/corpus.jsonl");

    if !corpus_path.exists() {
        return Err(AppError::Config(format!(
            "No corpus found for profile '{}'. Run: writer learn <files>",
            cfg.active_profile
        )));
    }

    // Load fingerprint
    let fp_path = profile_dir.join("fingerprint.json");
    let fingerprint = if fp_path.exists() {
        let fp_json = std::fs::read_to_string(&fp_path)?;
        serde_json::from_str::<StylometricFingerprint>(&fp_json)
            .map_err(|e| AppError::Config(format!("Invalid fingerprint.json: {e}")))?
    } else {
        return Err(AppError::Config(
            "No fingerprint found. Run: writer learn <files>".to_string(),
        ));
    };

    // Load prompt suite
    if !suite_path.exists() {
        return Err(AppError::InvalidInput(format!(
            "Prompt suite not found: {}",
            suite_path.display()
        )));
    }

    // Load canon lexicon
    let leakage_lexicon = load_canon_lexicon(&profile_dir);

    // Parse model ID
    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model: {e}")))?;

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Create ablation workspace
    let workspace = output_dir.join("workspace");
    std::fs::create_dir_all(&workspace)?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!("{} Starting format ablation", ">".blue());
        println!(
            "  {} training formats x {} inference modes = {} combinations",
            ABLATION_CONFIGS.len(),
            INFERENCE_MODES.len(),
            ABLATION_CONFIGS.len() * INFERENCE_MODES.len()
        );
        println!("  {} steps per adapter, {} seeds per prompt", steps, seeds);
        println!();
    }

    let mut all_results: Vec<AblationResult> = Vec::new();

    // Phase 1: Train adapters (unless eval_only)
    let mut trained_adapters: Vec<(AblationConfig, PathBuf, f32)> = Vec::new();

    for ablation_cfg in ABLATION_CONFIGS {
        let adapter_dir = workspace.join(ablation_cfg.name).join("adapters");

        if eval_only {
            // Check if adapter exists
            if !adapter_dir.join("adapters.safetensors").exists() {
                return Err(AppError::InvalidInput(format!(
                    "Adapter not found for '{}' at {}. Run without --eval-only first.",
                    ablation_cfg.name,
                    adapter_dir.display()
                )));
            }
            // Read loss from adapter_config.json if available, otherwise use 0.0
            let loss = read_adapter_loss(&adapter_dir).unwrap_or(0.0);
            trained_adapters.push((*ablation_cfg, adapter_dir, loss));
            continue;
        }

        if !ctx.format.is_json() {
            use owo_colors::OwoColorize;
            println!(
                "{} Training {} adapter ({} steps)...",
                ">".blue(),
                ablation_cfg.name,
                steps
            );
        }

        // Prepare training data in the correct format
        let data_dir = workspace.join(ablation_cfg.name).join("training_data");
        std::fs::create_dir_all(&data_dir)?;

        let (n_train, n_valid) =
            mlx_tune::prepare_training_data(&corpus_path, &data_dir, 0.1, ablation_cfg.format)
                .map_err(|e| AppError::Transient(e.to_string()))?;

        if !ctx.format.is_json() {
            println!("    {} train + {} valid samples", n_train, n_valid);
        }

        // Create adapter output directory
        std::fs::create_dir_all(&adapter_dir)?;

        let lora_config = LoraConfig {
            profile: ablation_cfg.name.to_string(),
            base_model: model_id.clone(),
            dataset_dir: data_dir,
            adapter_out: adapter_dir.clone(),
            rank: cfg.training.rank,
            alpha: cfg.training.alpha,
            learning_rate: cfg.training.learning_rate,
            batch_size: cfg.training.batch_size,
            max_steps: steps,
            max_seq_len: cfg.training.max_seq_len,
            mask_prompt: ablation_cfg.mask_prompt,
        };

        // Train
        let backend = MlxTuneBackend::new().map_err(|e| AppError::Transient(e.to_string()))?;

        let progress_fn = |p: writer_cli::backends::training::config::TrainingProgress| {
            eprint!(
                "\r    step {}/{} | loss: {:.4}",
                p.step, p.total_steps, p.loss
            );
        };

        let artefact = backend
            .train_lora(lora_config, &progress_fn)
            .await
            .map_err(|e| AppError::Transient(e.to_string()))?;

        if !ctx.format.is_json() {
            eprintln!(); // newline after progress
            println!("    final loss: {:.4}", artefact.final_loss);
        }

        trained_adapters.push((*ablation_cfg, adapter_dir, artefact.final_loss));
    }

    // Phase 2: Evaluate each adapter under each inference mode
    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!();
        println!("{} Running evaluations...", ">".blue());
    }

    // Ablation-specific decoding: n_candidates=1, max_attempts=1, fixed params
    // We're comparing formats, not ranking candidates.
    let ablation_decoding = writer_cli::config::DecodingConfig {
        n_candidates: 1,
        max_tokens: cfg.decoding.max_tokens,
        max_attempts: Some(1),
        contrastive_alpha: 0.0,
        banned_word_bias: cfg.decoding.banned_word_bias,
        preferred_word_bias: cfg.decoding.preferred_word_bias,
        kv_quant: cfg.decoding.kv_quant.clone(),
    };

    // Load prompt suite
    let suite_content = std::fs::read_to_string(&suite_path)?;
    let suite: PromptSuite = serde_yaml::from_str(&suite_content)
        .map_err(|e| AppError::InvalidInput(format!("Invalid YAML: {e}")))?;

    // Progress tracking
    let total_gens = ABLATION_CONFIGS.len() * INFERENCE_MODES.len() * suite.prompts.len() * seeds as usize;
    let progress_path = output_dir.join("progress.log");
    let mut completed_gens: usize = 0;
    let eval_start = std::time::Instant::now();

    // Write initial progress
    write_progress(&progress_path, 0, total_gens, 0.0, "starting evaluation", "")?;

    // Create MLX backend
    let mlx_backend = MlxBackend::new().map_err(|e| AppError::Transient(e.to_string()))?;
    let handle = mlx_backend
        .load_model(&model_id)
        .await
        .map_err(|e| AppError::Transient(e.to_string()))?;

    for (ablation_cfg, adapter_path, final_loss) in &trained_adapters {
        let adapter = writer_cli::backends::types::AdapterRef::new(
            ablation_cfg.name.to_string(),
            adapter_path.clone(),
        );

        for inf_mode in INFERENCE_MODES {
            let combo_name = format!("{} + {}", ablation_cfg.name, inf_mode.name);
            if !ctx.format.is_json() {
                eprintln!(
                    "  evaluating: {}",
                    combo_name,
                );
            }

            let mut records: Vec<EvalRecord> = Vec::new();

            for (pi, prompt_entry) in suite.prompts.iter().enumerate() {
                for seed in 0..seeds {
                    let system = if inf_mode.raw {
                        None
                    } else if fingerprint.word_count > 0 {
                        Some(prompts::system_prompt(&fingerprint))
                    } else {
                        None
                    };

                    let write_prompt = if inf_mode.raw {
                        prompt_entry.text.clone()
                    } else {
                        prompts::write_prompt(&prompt_entry.text)
                    };

                    let prompt_mode = if inf_mode.raw { Some("raw") } else { None };

                    // Deterministic seed: same prompt+seed pair across all combos
                    let rng_seed = (pi as u64) * 1000 + seed as u64 + 42;

                    let gen_start = std::time::Instant::now();

                    let result = decoding::run(
                        &mlx_backend,
                        &handle,
                        &model_id,
                        &fingerprint,
                        &ablation_decoding,
                        &write_prompt,
                        system.as_deref(),
                        Some(&adapter),
                        prompt_mode,
                        Some(rng_seed),
                    )
                    .await;

                    completed_gens += 1;
                    let gen_elapsed = gen_start.elapsed().as_secs();
                    let total_elapsed = eval_start.elapsed().as_secs_f64();
                    let avg_per_gen = if completed_gens > 0 {
                        total_elapsed / completed_gens as f64
                    } else {
                        0.0
                    };
                    let remaining = (total_gens - completed_gens) as f64 * avg_per_gen;

                    let status = format!(
                        "prompt {}/{} seed {}/{}",
                        pi + 1,
                        suite.prompts.len(),
                        seed + 1,
                        seeds,
                    );

                    write_progress(
                        &progress_path,
                        completed_gens,
                        total_gens,
                        remaining,
                        &combo_name,
                        &status,
                    )?;

                    if let Ok(r) = &result {
                        let punct = PunctuationStats::compute(&r.text);
                        let read = ReadabilityStats::compute(&r.text);
                        let canon_leakage =
                            compute_canon_leakage(&r.text, &prompt_entry.text, &leakage_lexicon);

                        if !ctx.format.is_json() {
                            eprintln!(
                                "    [{}/{}] {}s dist={:.3} leak={:.3} | ETA {:.0}m",
                                completed_gens,
                                total_gens,
                                gen_elapsed,
                                r.distance,
                                canon_leakage,
                                remaining / 60.0,
                            );
                        }

                        records.push(EvalRecord {
                            style_distance: r.distance,
                            fk_grade: read.flesch_kincaid_grade,
                            questions_per_1k: punct.questions_per_1k,
                            exclamations_per_1k: punct.exclamations_per_1k,
                            canon_leakage_score: canon_leakage,
                        });
                    } else if !ctx.format.is_json() {
                        eprintln!(
                            "    [{}/{}] FAILED | ETA {:.0}m",
                            completed_gens, total_gens, remaining / 60.0,
                        );
                    }
                }
            }

            if records.is_empty() {
                if !ctx.format.is_json() {
                    eprintln!("no successful generations");
                }
                continue;
            }

            // Compute summary stats
            let n = records.len() as f64;
            let mut distances: Vec<f64> = records.iter().map(|r| r.style_distance).collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if distances.len() % 2 == 0 {
                let mid = distances.len() / 2;
                (distances[mid - 1] + distances[mid]) / 2.0
            } else {
                distances[distances.len() / 2]
            };

            let result = AblationResult {
                training_format: ablation_cfg.name.to_string(),
                inference_mode: inf_mode.name.to_string(),
                mean_style_distance: records.iter().map(|r| r.style_distance).sum::<f64>() / n,
                median_style_distance: median,
                mean_fk_grade: records.iter().map(|r| r.fk_grade).sum::<f64>() / n,
                mean_questions_per_1k: records.iter().map(|r| r.questions_per_1k).sum::<f64>() / n,
                mean_exclamations_per_1k: records
                    .iter()
                    .map(|r| r.exclamations_per_1k)
                    .sum::<f64>()
                    / n,
                mean_canon_leakage: records.iter().map(|r| r.canon_leakage_score).sum::<f64>() / n,
                generations: records.len(),
                final_loss: *final_loss,
            };

            if !ctx.format.is_json() {
                println!(
                    "dist={:.3} leak={:.3} ({} gens)",
                    result.mean_style_distance, result.mean_canon_leakage, result.generations
                );
            }

            all_results.push(result);
        }
    }

    // Find best combination (lowest style distance with low leakage)
    let best = all_results
        .iter()
        .min_by(|a, b| {
            // Primary: style distance. Secondary: canon leakage.
            let score_a = a.mean_style_distance + a.mean_canon_leakage * 0.5;
            let score_b = b.mean_style_distance + b.mean_canon_leakage * 0.5;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| format!("{} + {}", r.training_format, r.inference_mode))
        .unwrap_or_else(|| "none".to_string());

    let best_dist = all_results
        .iter()
        .map(|r| r.mean_style_distance)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    let summary = AblationSummary {
        configs_tested: ABLATION_CONFIGS.len(),
        inference_modes: INFERENCE_MODES.len(),
        total_combinations: all_results.len(),
        best_combination: best.clone(),
        best_style_distance: best_dist,
        results: all_results,
    };

    // Write results
    let summary_path = output_dir.join("ablation_summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|e| AppError::Transient(format!("JSON serialization: {e}")))?;
    std::fs::write(&summary_path, &summary_json)?;

    // Write comparison table as CSV
    let csv_path = output_dir.join("ablation_comparison.csv");
    write_comparison_csv(&csv_path, &summary.results)?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!();
        println!("{} Ablation complete", "+".green());
        println!();
        println!("  {} combinations tested", summary.total_combinations);
        println!(
            "  best: {} (dist={:.3})",
            summary.best_combination.bold(),
            summary.best_style_distance
        );
        println!();
        println!("  Comparison table:");
        println!();
        print_comparison_table(&summary.results);
        println!();
        println!("  results: {}", output_dir.display().to_string().dimmed());
    } else {
        crate::output::print_success_or(ctx, &summary, |_| {});
    }

    Ok(())
}

// ── Helper types and functions ────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct PromptSuite {
    #[serde(default)]
    name: String,
    prompts: Vec<PromptEntry>,
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct PromptEntry {
    text: String,
    #[serde(default)]
    category: String,
}

#[derive(Debug)]
struct EvalRecord {
    style_distance: f64,
    fk_grade: f64,
    questions_per_1k: f64,
    exclamations_per_1k: f64,
    canon_leakage_score: f64,
}

fn load_canon_lexicon(profile_dir: &Path) -> Vec<String> {
    let path = profile_dir.join("canon_lexicon.txt");
    if !path.exists() {
        return Vec::new();
    }
    std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .map(|l| l.trim().to_lowercase())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

fn compute_canon_leakage(output: &str, prompt: &str, lexicon: &[String]) -> f64 {
    if lexicon.is_empty() {
        return 0.0;
    }

    let output_lower = output.to_lowercase();
    let prompt_lower = prompt.to_lowercase();

    let mut leaked = 0;
    let mut checkable = 0;

    for term in lexicon {
        if prompt_lower.contains(term.as_str()) {
            continue;
        }
        checkable += 1;
        if output_lower.contains(term.as_str()) {
            leaked += 1;
        }
    }

    if checkable == 0 {
        return 0.0;
    }

    leaked as f64 / checkable as f64
}

/// Write a machine-readable progress file that can be tailed/polled externally.
/// Format: single JSON object, overwritten on every generation.
fn write_progress(
    path: &Path,
    completed: usize,
    total: usize,
    eta_seconds: f64,
    combo: &str,
    status: &str,
) -> Result<(), AppError> {
    let pct = if total > 0 {
        (completed as f64 / total as f64 * 100.0).round()
    } else {
        0.0
    };
    let eta_min = (eta_seconds / 60.0).round();
    let content = format!(
        "{{\n  \"completed\": {},\n  \"total\": {},\n  \"percent\": {},\n  \"eta_minutes\": {},\n  \"current_combo\": \"{}\",\n  \"status\": \"{}\"\n}}\n",
        completed, total, pct, eta_min, combo, status,
    );
    std::fs::write(path, content).map_err(AppError::Io)
}

fn read_adapter_loss(adapter_dir: &Path) -> Option<f32> {
    let config_path = adapter_dir.join("adapter_config.json");
    if !config_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("final_loss")?.as_f64().map(|f| f as f32)
}

fn write_comparison_csv(path: &Path, results: &[AblationResult]) -> Result<(), AppError> {
    let mut out = String::new();
    out.push_str("training_format,inference_mode,mean_dist,median_dist,fk_grade,questions_1k,exclamations_1k,canon_leakage,final_loss,generations\n");

    for r in results {
        out.push_str(&format!(
            "{},{},{:.4},{:.4},{:.1},{:.1},{:.1},{:.4},{:.4},{}\n",
            r.training_format,
            r.inference_mode,
            r.mean_style_distance,
            r.median_style_distance,
            r.mean_fk_grade,
            r.mean_questions_per_1k,
            r.mean_exclamations_per_1k,
            r.mean_canon_leakage,
            r.final_loss,
            r.generations,
        ));
    }

    std::fs::write(path, out).map_err(AppError::Io)
}

fn print_comparison_table(results: &[AblationResult]) {
    use owo_colors::OwoColorize;

    println!(
        "  {:20} {:10} {:>8} {:>8} {:>8}",
        "format".bold(),
        "mode".bold(),
        "dist".bold(),
        "leak".bold(),
        "loss".bold()
    );
    println!("  {}", "-".repeat(58));

    for r in results {
        let dist_color = if r.mean_style_distance < 0.25 {
            "\x1b[32m" // green
        } else if r.mean_style_distance < 0.30 {
            "\x1b[33m" // yellow
        } else {
            "\x1b[31m" // red
        };

        let leak_color = if r.mean_canon_leakage < 0.05 {
            "\x1b[32m"
        } else if r.mean_canon_leakage < 0.10 {
            "\x1b[33m"
        } else {
            "\x1b[31m"
        };

        println!(
            "  {:20} {:10} {}{:>8.3}\x1b[0m {}{:>8.3}\x1b[0m {:>8.3}",
            r.training_format,
            r.inference_mode,
            dist_color,
            r.mean_style_distance,
            leak_color,
            r.mean_canon_leakage,
            r.final_loss,
        );
    }
}
