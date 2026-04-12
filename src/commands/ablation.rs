//! `writer ablation` — automated format ablation experiment.
//!
//! Trains 3 adapters (chat, completions+mask, text) on the same corpus split,
//! then evaluates each under multiple inference modes (wrapped, raw).
//! Produces a comparison report to guide format selection.
use std::path::{Path, PathBuf};

use serde::Serialize;

use writer_cli::backends::inference::mlx_worker::{MlxWorker, WorkerRequest};
use writer_cli::backends::training::TrainingBackend;
use writer_cli::backends::training::config::LoraConfig;
use writer_cli::backends::training::mlx_tune::{self, MlxTuneBackend};
use writer_cli::backends::types::ModelId;
use writer_cli::config::DatasetFormat;
use writer_cli::decoding::logit_bias;
use writer_cli::decoding::prompts;
use writer_cli::stylometry::features::punctuation::PunctuationStats;
use writer_cli::stylometry::features::readability::ReadabilityStats;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;
use writer_cli::stylometry::scoring;

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

/// Per-generation record — written to per_generation.jsonl
#[derive(Debug, Clone, Serialize)]
struct GenerationRecord {
    training_format: String,
    inference_mode: String,
    prompt: String,
    category: String,
    seed: u16,
    rng_seed: u64,
    status: String, // "ok" or "failed"
    error: Option<String>,
    text: Option<String>,
    style_distance: Option<f64>,
    base_voice_distance: Option<f64>,
    fk_grade: Option<f64>,
    questions_per_1k: Option<f64>,
    exclamations_per_1k: Option<f64>,
    terminal_punct_dist: Option<f64>,
    structural_punct_dist: Option<f64>,
    sentence_length_dist: Option<f64>,
    function_word_cos: Option<f64>,
    ngram_cos: Option<f64>,
    readability_diff: Option<f64>,
    richness_diff: Option<f64>,
    slop_score: Option<f64>,
    slop_multiplier: Option<f64>,
    canon_leakage_score: Option<f64>,
    leaked_terms: Option<Vec<String>>,
    word_count: Option<usize>,
    sentence_count: Option<usize>,
    elapsed_ms: Option<u64>,
}

/// Bucket-level statistics
#[derive(Debug, Clone, Serialize)]
struct BucketStats {
    bucket: String,
    count: usize,
    failed: usize,
    mean_style_distance: f64,
    median_style_distance: f64,
    variance_style_distance: f64,
    mean_canon_leakage: f64,
    worst_canon_leakage: f64,
    mean_questions_per_1k: f64,
    mean_exclamations_per_1k: f64,
    mean_terminal_punct_dist: f64,
    mean_structural_punct_dist: f64,
    mean_fk_grade: f64,
}

/// Per-combination summary with bucket breakdowns
#[derive(Debug, Serialize)]
struct ComboResult {
    training_format: String,
    inference_mode: String,
    final_loss: f32,
    total: usize,
    succeeded: usize,
    failed: usize,
    overall: BucketStats,
    buckets: std::collections::BTreeMap<String, BucketStats>,
}

#[derive(Debug, Serialize)]
struct AblationSummary {
    configs_tested: usize,
    inference_modes: usize,
    total_combinations: usize,
    winner: WinnerReport,
    results: Vec<ComboResult>,
}

#[derive(Debug, Serialize)]
struct WinnerReport {
    combination: String,
    gate_passed: String,
    off_domain_leakage: f64,
    overall_style_distance: f64,
    reasoning: Vec<String>,
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

    let mut all_results: Vec<ComboResult> = Vec::new();
    let mut all_gen_records: Vec<GenerationRecord> = Vec::new();

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

    // Load prompt suite
    let suite_content = std::fs::read_to_string(&suite_path)?;
    let suite: PromptSuite = serde_yaml::from_str(&suite_content)
        .map_err(|e| AppError::InvalidInput(format!("Invalid YAML: {e}")))?;

    // Build logit bias once (same fingerprint for all combos)
    let bias = logit_bias::from_fingerprint(&fingerprint, &cfg.decoding);

    // Progress tracking
    let total_gens =
        trained_adapters.len() * INFERENCE_MODES.len() * suite.prompts.len() * seeds as usize;
    let progress_path = output_dir.join("progress.log");
    let mut completed_gens: usize = 0;
    let eval_start = std::time::Instant::now();
    write_progress(
        &progress_path,
        0,
        total_gens,
        0.0,
        "starting evaluation",
        "",
    )?;

    // Evaluate: one persistent worker per adapter (model loaded once per adapter)
    for (ablation_cfg, adapter_path, final_loss) in &trained_adapters {
        if !ctx.format.is_json() {
            eprintln!("  spawning worker for {} adapter...", ablation_cfg.name);
        }

        let mut worker = MlxWorker::spawn(&model_id, Some(adapter_path.as_path()))
            .await
            .map_err(|e| AppError::Transient(format!("Worker spawn failed: {e}")))?;

        for inf_mode in INFERENCE_MODES {
            let combo_name = format!("{} + {}", ablation_cfg.name, inf_mode.name);
            if !ctx.format.is_json() {
                eprintln!("  evaluating: {combo_name}");
            }

            let mut gen_records: Vec<GenerationRecord> = Vec::new();

            for (pi, prompt_entry) in suite.prompts.iter().enumerate() {
                let bucket = normalize_bucket(&prompt_entry.category);

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

                    let prompt_mode_str = if inf_mode.raw { "raw" } else { "chat" };
                    let rng_seed = (pi as u64) * 1000 + seed as u64 + 42;
                    let gen_start = std::time::Instant::now();

                    let req = WorkerRequest {
                        prompt: write_prompt,
                        system_prompt: system,
                        prompt_mode: prompt_mode_str.to_string(),
                        max_tokens: cfg.decoding.max_tokens,
                        temperature: cfg.inference.temperature,
                        top_p: 0.92,
                        repetition_penalty: cfg.decoding.repetition_penalty,
                        seed: Some(rng_seed),
                        logit_bias: bias.clone(),
                    };

                    let result = worker.generate(&req).await;

                    completed_gens += 1;
                    let gen_elapsed_ms = gen_start.elapsed().as_millis() as u64;
                    let total_elapsed = eval_start.elapsed().as_secs_f64();
                    let avg_per_gen = if completed_gens > 0 {
                        total_elapsed / completed_gens as f64
                    } else {
                        0.0
                    };
                    let remaining = (total_gens - completed_gens) as f64 * avg_per_gen;

                    write_progress(
                        &progress_path,
                        completed_gens,
                        total_gens,
                        remaining,
                        &combo_name,
                        &format!(
                            "prompt {}/{} seed {}/{}",
                            pi + 1,
                            suite.prompts.len(),
                            seed + 1,
                            seeds
                        ),
                    )?;

                    let record = match result {
                        Ok(events) => {
                            // Extract text from generation events
                            let text = events.iter().find_map(|e| {
                                if let writer_cli::backends::inference::response::GenerationEvent::Done { full_text, .. } = e {
                                    Some(full_text.clone())
                                } else {
                                    None
                                }
                            }).unwrap_or_default();

                            if text.is_empty() {
                                GenerationRecord {
                                    training_format: ablation_cfg.name.to_string(),
                                    inference_mode: inf_mode.name.to_string(),
                                    prompt: prompt_entry.text.clone(),
                                    category: bucket.clone(),
                                    seed,
                                    rng_seed,
                                    status: "failed".to_string(),
                                    error: Some("Empty generation".to_string()),
                                    text: None,
                                    style_distance: None,
                                    base_voice_distance: None,
                                    fk_grade: None,
                                    questions_per_1k: None,
                                    exclamations_per_1k: None,
                                    terminal_punct_dist: None,
                                    structural_punct_dist: None,
                                    sentence_length_dist: None,
                                    function_word_cos: None,
                                    ngram_cos: None,
                                    readability_diff: None,
                                    richness_diff: None,
                                    slop_score: None,
                                    slop_multiplier: None,
                                    canon_leakage_score: None,
                                    leaked_terms: None,
                                    word_count: None,
                                    sentence_count: None,
                                    elapsed_ms: Some(gen_elapsed_ms),
                                }
                            } else {
                                let punct = PunctuationStats::compute(&text);
                                let read = ReadabilityStats::compute(&text);
                                let dist_report = scoring::distance(&text, &fingerprint);
                                let leakage = compute_canon_leakage(
                                    &text,
                                    &prompt_entry.text,
                                    &leakage_lexicon,
                                );
                                let wc = text.split_whitespace().count();
                                let sc = text.split_terminator(['.', '!', '?']).count();

                                if !ctx.format.is_json() {
                                    eprintln!(
                                        "    [{}/{}] {:.0}s dist={:.3} leak={:.3} | ETA {:.0}m",
                                        completed_gens,
                                        total_gens,
                                        gen_elapsed_ms as f64 / 1000.0,
                                        dist_report.overall,
                                        leakage.score,
                                        remaining / 60.0
                                    );
                                }

                                GenerationRecord {
                                    training_format: ablation_cfg.name.to_string(),
                                    inference_mode: inf_mode.name.to_string(),
                                    prompt: prompt_entry.text.clone(),
                                    category: bucket.clone(),
                                    seed,
                                    rng_seed,
                                    status: "ok".to_string(),
                                    error: None,
                                    text: Some(text),
                                    style_distance: Some(dist_report.overall),
                                    base_voice_distance: Some(dist_report.base_voice_distance),
                                    fk_grade: Some(read.flesch_kincaid_grade),
                                    questions_per_1k: Some(punct.questions_per_1k),
                                    exclamations_per_1k: Some(punct.exclamations_per_1k),
                                    terminal_punct_dist: Some(dist_report.terminal_punct_dist),
                                    structural_punct_dist: Some(dist_report.structural_punct_dist),
                                    sentence_length_dist: Some(dist_report.sentence_length_dist),
                                    function_word_cos: Some(dist_report.function_word_cos),
                                    ngram_cos: Some(dist_report.ngram_cos),
                                    readability_diff: Some(dist_report.readability_diff),
                                    richness_diff: Some(dist_report.richness_diff),
                                    slop_score: Some(dist_report.slop_score),
                                    slop_multiplier: Some(dist_report.slop_multiplier),
                                    canon_leakage_score: Some(leakage.score),
                                    leaked_terms: Some(leakage.leaked_terms),
                                    word_count: Some(wc),
                                    sentence_count: Some(sc),
                                    elapsed_ms: Some(gen_elapsed_ms),
                                }
                            }
                        }
                        Err(e) => {
                            if !ctx.format.is_json() {
                                eprintln!(
                                    "    [{}/{}] FAILED: {} | ETA {:.0}m",
                                    completed_gens,
                                    total_gens,
                                    e,
                                    remaining / 60.0
                                );
                            }
                            GenerationRecord {
                                training_format: ablation_cfg.name.to_string(),
                                inference_mode: inf_mode.name.to_string(),
                                prompt: prompt_entry.text.clone(),
                                category: bucket.clone(),
                                seed,
                                rng_seed,
                                status: "failed".to_string(),
                                error: Some(e.to_string()),
                                text: None,
                                style_distance: None,
                                base_voice_distance: None,
                                fk_grade: None,
                                questions_per_1k: None,
                                exclamations_per_1k: None,
                                terminal_punct_dist: None,
                                structural_punct_dist: None,
                                sentence_length_dist: None,
                                function_word_cos: None,
                                ngram_cos: None,
                                readability_diff: None,
                                richness_diff: None,
                                slop_score: None,
                                slop_multiplier: None,
                                canon_leakage_score: None,
                                leaked_terms: None,
                                word_count: None,
                                sentence_count: None,
                                elapsed_ms: Some(gen_elapsed_ms),
                            }
                        }
                    };

                    all_gen_records.push(record.clone());
                    gen_records.push(record);
                }
            }

            let combo_result =
                compute_combo_result(ablation_cfg.name, inf_mode.name, *final_loss, &gen_records);

            if !ctx.format.is_json() {
                let o = &combo_result.overall;
                println!(
                    "  {} + {} → dist={:.3} leak={:.3} ({}/{} ok)",
                    ablation_cfg.name,
                    inf_mode.name,
                    o.mean_style_distance,
                    o.mean_canon_leakage,
                    combo_result.succeeded,
                    combo_result.total,
                );
            }

            all_results.push(combo_result);
        }

        // Worker is dropped here — child process exits when stdin closes
    }

    // ── Write per-generation records ──────────────────────────────────────
    let gen_path = output_dir.join("per_generation.jsonl");
    let mut gen_out = String::new();
    for rec in &all_gen_records {
        if let Ok(line) = serde_json::to_string(rec) {
            gen_out.push_str(&line);
            gen_out.push('\n');
        }
    }
    std::fs::write(&gen_path, &gen_out)?;

    // ── Hard-gate winner selection ────────────────────────────────────────
    let winner = select_winner(&all_results);

    let summary = AblationSummary {
        configs_tested: ABLATION_CONFIGS.len(),
        inference_modes: INFERENCE_MODES.len(),
        total_combinations: all_results.len(),
        winner,
        results: all_results,
    };

    let summary_path = output_dir.join("ablation_summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|e| AppError::Transient(format!("JSON serialization: {e}")))?;
    std::fs::write(&summary_path, &summary_json)?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!();
        println!("{} Ablation complete", "+".green());
        println!();
        println!("  winner: {}", summary.winner.combination.bold());
        println!("  gate: {}", summary.winner.gate_passed);
        for reason in &summary.winner.reasoning {
            println!("    - {reason}");
        }
        println!();
        println!("  results: {}", output_dir.display().to_string().dimmed());
        println!("  per-gen: {}", gen_path.display().to_string().dimmed());
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
struct PromptEntry {
    text: String,
    #[serde(default)]
    category: String,
}

/// Leakage result with matched terms for diagnostics
struct LeakageResult {
    score: f64,
    leaked_terms: Vec<String>,
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

fn compute_canon_leakage(output: &str, prompt: &str, lexicon: &[String]) -> LeakageResult {
    if lexicon.is_empty() {
        return LeakageResult {
            score: 0.0,
            leaked_terms: Vec::new(),
        };
    }

    let output_lower = output.to_lowercase();
    let prompt_lower = prompt.to_lowercase();

    let mut leaked_terms = Vec::new();
    let mut checkable = 0;

    for term in lexicon {
        // Skip terms present in the prompt (using same whole-word matching)
        if contains_whole(term, &prompt_lower) {
            continue;
        }
        checkable += 1;
        if contains_whole(term, &output_lower) {
            leaked_terms.push(term.clone());
        }
    }

    let score = if checkable == 0 {
        0.0
    } else {
        leaked_terms.len() as f64 / checkable as f64
    };

    LeakageResult {
        score,
        leaked_terms,
    }
}

/// Check if `needle` appears in `haystack` at a Unicode word boundary.
/// Prevents "art" matching inside "article" etc.
fn contains_whole(needle: &str, haystack: &str) -> bool {
    for (start, _) in haystack.match_indices(needle) {
        let end = start + needle.len();
        let before_ok = start == 0
            || !haystack[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric());
        let after_ok = end >= haystack.len()
            || !haystack[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
    }
    false
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

/// Normalize prompt category to standard buckets.
/// Anything not canon-adjacent/longform/shortform → off-domain.
fn normalize_bucket(category: &str) -> String {
    match category {
        "canon-adjacent" => "canon-adjacent".to_string(),
        "longform" => "longform".to_string(),
        "shortform" => "shortform".to_string(),
        _ => "off-domain".to_string(),
    }
}

/// Compute per-combination stats with bucket breakdowns.
fn compute_combo_result(
    format_name: &str,
    mode_name: &str,
    final_loss: f32,
    records: &[GenerationRecord],
) -> ComboResult {
    let succeeded = records.iter().filter(|r| r.status == "ok").count();
    let failed = records.iter().filter(|r| r.status == "failed").count();

    // Overall stats from successful records
    let ok_records: Vec<&GenerationRecord> = records.iter().filter(|r| r.status == "ok").collect();
    let overall = compute_bucket_stats("overall", &ok_records, failed);

    // Per-bucket stats — always include all standard buckets
    let mut buckets = std::collections::BTreeMap::new();
    let standard_buckets = ["off-domain", "canon-adjacent", "longform", "shortform"];
    let mut bucket_names: std::collections::BTreeSet<String> =
        records.iter().map(|r| r.category.clone()).collect();
    for b in &standard_buckets {
        bucket_names.insert(b.to_string());
    }

    for bucket_name in &bucket_names {
        let bucket_ok: Vec<&GenerationRecord> = ok_records
            .iter()
            .filter(|r| r.category == *bucket_name)
            .copied()
            .collect();
        let bucket_failed = records
            .iter()
            .filter(|r| r.category == *bucket_name && r.status == "failed")
            .count();
        buckets.insert(
            bucket_name.clone(),
            compute_bucket_stats(bucket_name, &bucket_ok, bucket_failed),
        );
    }

    ComboResult {
        training_format: format_name.to_string(),
        inference_mode: mode_name.to_string(),
        final_loss,
        total: records.len(),
        succeeded,
        failed,
        overall,
        buckets,
    }
}

fn compute_bucket_stats(name: &str, records: &[&GenerationRecord], failed: usize) -> BucketStats {
    if records.is_empty() {
        // All-failure: use f64::MAX for distance/leakage so these can't win
        return BucketStats {
            bucket: name.to_string(),
            count: 0,
            failed,
            mean_style_distance: f64::MAX,
            median_style_distance: f64::MAX,
            variance_style_distance: 0.0,
            mean_canon_leakage: f64::MAX,
            worst_canon_leakage: f64::MAX,
            mean_questions_per_1k: 0.0,
            mean_exclamations_per_1k: 0.0,
            mean_terminal_punct_dist: f64::MAX,
            mean_structural_punct_dist: f64::MAX,
            mean_fk_grade: 0.0,
        };
    }

    let n = records.len() as f64;

    let dists: Vec<f64> = records.iter().filter_map(|r| r.style_distance).collect();
    let mean_dist = dists.iter().sum::<f64>() / n;

    let mut sorted_dists = dists.clone();
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_dist = if sorted_dists.len() % 2 == 0 {
        let m = sorted_dists.len() / 2;
        (sorted_dists[m - 1] + sorted_dists[m]) / 2.0
    } else {
        sorted_dists[sorted_dists.len() / 2]
    };

    let variance_dist = dists.iter().map(|d| (d - mean_dist).powi(2)).sum::<f64>() / n;

    let leakages: Vec<f64> = records
        .iter()
        .filter_map(|r| r.canon_leakage_score)
        .collect();
    let mean_leak = leakages.iter().sum::<f64>() / n;
    let worst_leak = leakages.iter().copied().fold(0.0f64, f64::max);

    BucketStats {
        bucket: name.to_string(),
        count: records.len(),
        failed,
        mean_style_distance: mean_dist,
        median_style_distance: median_dist,
        variance_style_distance: variance_dist,
        mean_canon_leakage: mean_leak,
        worst_canon_leakage: worst_leak,
        mean_questions_per_1k: records
            .iter()
            .filter_map(|r| r.questions_per_1k)
            .sum::<f64>()
            / n,
        mean_exclamations_per_1k: records
            .iter()
            .filter_map(|r| r.exclamations_per_1k)
            .sum::<f64>()
            / n,
        mean_terminal_punct_dist: records
            .iter()
            .filter_map(|r| r.terminal_punct_dist)
            .sum::<f64>()
            / n,
        mean_structural_punct_dist: records
            .iter()
            .filter_map(|r| r.structural_punct_dist)
            .sum::<f64>()
            / n,
        mean_fk_grade: records.iter().filter_map(|r| r.fk_grade).sum::<f64>() / n,
    }
}

/// Hard-gate winner selection per the decision protocol:
/// Gate 1: lowest off-domain canon leakage (strict — only combos with 0 successful off-domain gens are excluded)
/// Gate 2: best structural voice distance (terminal + structural punct, lower = closer to author)
/// Gate 3: lowest overall style distance as tiebreaker
fn select_winner(results: &[ComboResult]) -> WinnerReport {
    if results.is_empty() {
        return WinnerReport {
            combination: "none".to_string(),
            gate_passed: "none".to_string(),
            off_domain_leakage: 0.0,
            overall_style_distance: 0.0,
            reasoning: vec!["No results to evaluate".to_string()],
        };
    }

    let mut reasoning = Vec::new();

    // Exclude combos with 0 successful generations
    let viable: Vec<usize> = results
        .iter()
        .enumerate()
        .filter(|(_, r)| r.succeeded > 0)
        .map(|(i, _)| i)
        .collect();

    if viable.is_empty() {
        return WinnerReport {
            combination: "none".to_string(),
            gate_passed: "all-failed".to_string(),
            off_domain_leakage: 0.0,
            overall_style_distance: 0.0,
            reasoning: vec!["All combinations failed to produce output".to_string()],
        };
    }

    reasoning.push(format!(
        "{} of {} combos have successful generations",
        viable.len(),
        results.len()
    ));

    // Gate 1: Sort by off-domain leakage, keep top half (strict lexicographic)
    let mut gate1_scored: Vec<(usize, f64)> = viable
        .iter()
        .map(|&i| {
            let leak = results[i]
                .buckets
                .get("off-domain")
                .filter(|b| b.count > 0)
                .map(|b| b.mean_canon_leakage)
                .unwrap_or(results[i].overall.mean_canon_leakage);
            (i, leak)
        })
        .collect();
    gate1_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Keep at most half, but at least 2 (or all if ≤ 2)
    let gate1_keep = gate1_scored
        .len()
        .min(gate1_scored.len() / 2 + 1)
        .max(2)
        .min(gate1_scored.len());
    let gate1_survivors: Vec<usize> = gate1_scored
        .iter()
        .take(gate1_keep)
        .map(|(i, _)| *i)
        .collect();

    for (i, leak) in &gate1_scored {
        let r = &results[*i];
        let survived = gate1_survivors.contains(i);
        reasoning.push(format!(
            "  G1: {} + {} leak={:.4} {}",
            r.training_format,
            r.inference_mode,
            leak,
            if survived { "PASS" } else { "ELIMINATED" }
        ));
    }

    // Gate 2: Sort by structural voice distance (terminal + structural punct dist, lower = better)
    let mut gate2_scored: Vec<(usize, f64)> = gate1_survivors
        .iter()
        .map(|&i| {
            let r = &results[i];
            let voice_dist =
                r.overall.mean_terminal_punct_dist + r.overall.mean_structural_punct_dist;
            (i, voice_dist)
        })
        .collect();
    gate2_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Keep at most half+1 or all if ≤ 2
    let gate2_keep = gate2_scored
        .len()
        .min(gate2_scored.len() / 2 + 1)
        .max(2)
        .min(gate2_scored.len());
    let gate2_survivors: Vec<usize> = gate2_scored
        .iter()
        .take(gate2_keep)
        .map(|(i, _)| *i)
        .collect();

    for (i, dist) in &gate2_scored {
        let r = &results[*i];
        let survived = gate2_survivors.contains(i);
        reasoning.push(format!(
            "  G2: {} + {} voice_dist={:.4} {}",
            r.training_format,
            r.inference_mode,
            dist,
            if survived { "PASS" } else { "ELIMINATED" }
        ));
    }

    // Gate 3: Lowest style distance as tiebreaker
    let mut gate3_scored: Vec<(usize, f64)> = gate2_survivors
        .iter()
        .map(|&i| (i, results[i].overall.mean_style_distance))
        .collect();
    gate3_scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let winner_idx = gate3_scored[0].0;
    let r = &results[winner_idx];
    let winner_leak = gate1_scored
        .iter()
        .find(|(i, _)| *i == winner_idx)
        .map(|(_, l)| *l)
        .unwrap_or(0.0);

    reasoning.push(format!(
        "  WINNER: {} + {} dist={:.3} leak={:.4}",
        r.training_format, r.inference_mode, r.overall.mean_style_distance, winner_leak
    ));

    WinnerReport {
        combination: format!("{} + {}", r.training_format, r.inference_mode),
        gate_passed: "all-gates".to_string(),
        off_domain_leakage: winner_leak,
        overall_style_distance: r.overall.mean_style_distance,
        reasoning,
    }
}
