//! `writer eval-style` — automated style evaluation harness.
//!
//! Runs a prompt suite through the generation pipeline at multiple seeds,
//! recording stylometric distance, sentence length stats, FK grade,
//! question/exclamation rates, prompt relevance, canon leakage, and
//! generation config for each output.
//!
//! Outputs JSON and CSV summaries for cross-model/adapter comparison.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use writer_cli::backends::inference::mlx::MlxBackend;
use writer_cli::backends::inference::ollama::OllamaBackend;
use writer_cli::backends::inference::InferenceBackend;
use writer_cli::backends::types::{AdapterRef, ModelId};
use writer_cli::decoding;
use writer_cli::decoding::prompts;
use writer_cli::stylometry::features::lengths;
use writer_cli::stylometry::features::punctuation::PunctuationStats;
use writer_cli::stylometry::features::readability::ReadabilityStats;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;

use crate::config;
use crate::error::AppError;
use crate::output::Ctx;

// ── Prompt suite ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct PromptSuite {
    /// Optional suite-level metadata
    #[serde(default)]
    name: String,
    prompts: Vec<PromptEntry>,
}

#[derive(Debug, Deserialize)]
struct PromptEntry {
    /// The prompt text
    text: String,
    /// Optional category tag for grouping in reports
    #[serde(default)]
    category: String,
}

// ── Per-generation result ─────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct EvalRecord {
    prompt: String,
    category: String,
    seed: u16,
    text: String,
    style_distance: f64,
    sentence_length_mean: f64,
    sentence_length_sd: f64,
    fk_grade: f64,
    questions_per_1k: f64,
    exclamations_per_1k: f64,
    canon_leakage_score: f64,
    // Generation config
    system_prompt_enabled: bool,
    prompt_wrapping_enabled: bool,
    raw_mode: bool,
    adapter_used: bool,
    n_candidates: u16,
    model: String,
}

// ── Summary ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct EvalSummary {
    suite_name: String,
    total_prompts: usize,
    seeds_per_prompt: u16,
    total_generations: usize,
    mean_style_distance: f64,
    median_style_distance: f64,
    mean_fk_grade: f64,
    mean_questions_per_1k: f64,
    mean_exclamations_per_1k: f64,
    mean_canon_leakage: f64,
    raw_mode: bool,
    adapter_used: bool,
    model: String,
}

pub async fn run(
    ctx: Ctx,
    suite_path: PathBuf,
    seeds: u16,
    adapter_override: Option<PathBuf>,
    raw: bool,
    output_dir: PathBuf,
) -> Result<(), AppError> {
    // Load prompt suite
    let suite_content = std::fs::read_to_string(&suite_path).map_err(|e| {
        AppError::InvalidInput(format!("Cannot read suite file {}: {e}", suite_path.display()))
    })?;
    let suite: PromptSuite = serde_yaml::from_str(&suite_content).map_err(|e| {
        AppError::InvalidInput(format!("Invalid YAML in {}: {e}", suite_path.display()))
    })?;

    if suite.prompts.is_empty() {
        return Err(AppError::InvalidInput(
            "Prompt suite contains no prompts".to_string(),
        ));
    }

    let cfg = config::load()?;
    let model_id: ModelId = cfg
        .base_model
        .parse()
        .map_err(|e| AppError::Config(format!("Invalid base_model: {e}")))?;

    let profile_dir = config::profiles_dir().join(&cfg.active_profile);

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

    // Detect adapter
    let adapter = if let Some(ref path) = adapter_override {
        if path.join("adapters.safetensors").exists() {
            Some(AdapterRef::new(cfg.active_profile.clone(), path.clone()))
        } else {
            return Err(AppError::InvalidInput(format!(
                "No adapters.safetensors in {}",
                path.display()
            )));
        }
    } else {
        detect_adapter(&profile_dir)
    };

    // Choose backend
    let (backend_box, backend_name): (Box<dyn InferenceBackend>, &str) = if adapter.is_some() {
        match MlxBackend::new() {
            Ok(mlx) => (Box::new(mlx), "mlx"),
            Err(_) => {
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

    // Load canon leakage lexicon if available
    let leakage_lexicon = load_canon_lexicon(&profile_dir);

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!(
            "{} Evaluating {} prompts x {} seeds = {} generations",
            ">".blue(),
            suite.prompts.len(),
            seeds,
            suite.prompts.len() * seeds as usize,
        );
        println!(
            "  backend: {}, adapter: {}, raw: {}",
            backend_name,
            adapter.is_some(),
            raw
        );
    }

    let mut records: Vec<EvalRecord> = Vec::new();

    for (pi, prompt_entry) in suite.prompts.iter().enumerate() {
        for seed in 0..seeds {
            let system = if raw {
                None
            } else if fingerprint.word_count > 0 {
                Some(prompts::system_prompt(&fingerprint))
            } else {
                None
            };

            let write_prompt = if raw {
                prompt_entry.text.clone()
            } else {
                prompts::write_prompt(&prompt_entry.text)
            };

            let prompt_mode = if raw { Some("raw") } else { None };

            let result = decoding::run(
                backend_box.as_ref(),
                &handle,
                &model_id,
                &fingerprint,
                &cfg.decoding,
                &write_prompt,
                system.as_deref(),
                adapter.as_ref(),
                prompt_mode,
            )
            .await;

            let (text, distance) = match result {
                Ok(r) => (r.text, r.distance),
                Err(e) => {
                    if !ctx.format.is_json() {
                        eprintln!(
                            "  warning: prompt {} seed {} failed: {e}",
                            pi + 1,
                            seed + 1
                        );
                    }
                    continue;
                }
            };

            // Compute metrics
            let sent_stats = lengths::sentence_lengths(&text);
            let punct = PunctuationStats::compute(&text);
            let read = ReadabilityStats::compute(&text);
            let canon_leakage = compute_canon_leakage(&text, &prompt_entry.text, &leakage_lexicon);

            let record = EvalRecord {
                prompt: prompt_entry.text.clone(),
                category: prompt_entry.category.clone(),
                seed,
                text,
                style_distance: distance,
                sentence_length_mean: sent_stats.mean,
                sentence_length_sd: sent_stats.sd,
                fk_grade: read.flesch_kincaid_grade,
                questions_per_1k: punct.questions_per_1k,
                exclamations_per_1k: punct.exclamations_per_1k,
                canon_leakage_score: canon_leakage,
                system_prompt_enabled: system.is_some(),
                prompt_wrapping_enabled: !raw,
                raw_mode: raw,
                adapter_used: adapter.is_some(),
                n_candidates: cfg.decoding.n_candidates,
                model: model_id.to_string(),
            };

            if !ctx.format.is_json() {
                eprint!(
                    "\r  [{}/{}] prompt {}/{} seed {}/{} — dist: {:.3}",
                    records.len() + 1,
                    suite.prompts.len() * seeds as usize,
                    pi + 1,
                    suite.prompts.len(),
                    seed + 1,
                    seeds,
                    record.style_distance,
                );
            }

            records.push(record);
        }
    }

    if !ctx.format.is_json() {
        eprintln!(); // newline after progress
    }

    // Write outputs
    std::fs::create_dir_all(&output_dir)?;

    // JSON
    let json_path = output_dir.join("eval_results.json");
    let json_content = serde_json::to_string_pretty(&records)
        .map_err(|e| AppError::Transient(format!("JSON serialization: {e}")))?;
    std::fs::write(&json_path, &json_content)?;

    // CSV
    let csv_path = output_dir.join("eval_results.csv");
    write_csv(&csv_path, &records)?;

    // Summary
    let summary = compute_summary(&suite.name, &records, seeds, raw, adapter.is_some(), &model_id);
    let summary_path = output_dir.join("eval_summary.json");
    let summary_json = serde_json::to_string_pretty(&summary)
        .map_err(|e| AppError::Transient(format!("JSON serialization: {e}")))?;
    std::fs::write(&summary_path, &summary_json)?;

    if !ctx.format.is_json() {
        use owo_colors::OwoColorize;
        println!("{} Evaluation complete", "+".green());
        println!("  {} generations", summary.total_generations);
        println!(
            "  mean style distance: {:.3}",
            summary.mean_style_distance
        );
        println!(
            "  median style distance: {:.3}",
            summary.median_style_distance
        );
        println!("  mean FK grade: {:.1}", summary.mean_fk_grade);
        println!(
            "  mean questions/1k: {:.1}, exclamations/1k: {:.1}",
            summary.mean_questions_per_1k, summary.mean_exclamations_per_1k
        );
        println!(
            "  mean canon leakage: {:.3}",
            summary.mean_canon_leakage
        );
        println!(
            "\n  results: {}",
            output_dir.display().to_string().dimmed()
        );
    } else {
        crate::output::print_success_or(ctx, &summary, |_| {});
    }

    Ok(())
}

fn detect_adapter(profile_dir: &Path) -> Option<AdapterRef> {
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

// ── Canon leakage ─────────────────────────────────────────────────────────

/// Load canon lexicon from profile directory.
/// File format: one term per line in `canon_lexicon.txt`.
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

/// Score canon leakage: fraction of canon terms that appear in the output
/// but are not present in the prompt.
fn compute_canon_leakage(output: &str, prompt: &str, lexicon: &[String]) -> f64 {
    if lexicon.is_empty() {
        return 0.0;
    }

    let output_lower = output.to_lowercase();
    let prompt_lower = prompt.to_lowercase();

    let mut leaked = 0;
    let mut checkable = 0;

    for term in lexicon {
        // Skip terms present in the prompt (they're expected)
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

// ── CSV writer ────────────────────────────────────────────────────────────

fn write_csv(path: &Path, records: &[EvalRecord]) -> Result<(), AppError> {
    let mut out = String::new();
    out.push_str("prompt,category,seed,style_distance,sentence_length_mean,sentence_length_sd,fk_grade,questions_per_1k,exclamations_per_1k,canon_leakage_score,system_prompt,prompt_wrapping,raw_mode,adapter,n_candidates,model\n");

    for r in records {
        // CSV-escape the prompt
        let prompt_escaped = r.prompt.replace('"', "\"\"");
        out.push_str(&format!(
            "\"{}\",\"{}\",{},{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{},{},{},{},{},{}\n",
            prompt_escaped,
            r.category,
            r.seed,
            r.style_distance,
            r.sentence_length_mean,
            r.sentence_length_sd,
            r.fk_grade,
            r.questions_per_1k,
            r.exclamations_per_1k,
            r.canon_leakage_score,
            r.system_prompt_enabled,
            r.prompt_wrapping_enabled,
            r.raw_mode,
            r.adapter_used,
            r.n_candidates,
            r.model,
        ));
    }

    std::fs::write(path, out).map_err(AppError::Io)
}

// ── Summary computation ───────────────────────────────────────────────────

fn compute_summary(
    suite_name: &str,
    records: &[EvalRecord],
    seeds: u16,
    raw: bool,
    adapter: bool,
    model_id: &ModelId,
) -> EvalSummary {
    let n = records.len() as f64;
    if records.is_empty() {
        return EvalSummary {
            suite_name: suite_name.to_string(),
            total_prompts: 0,
            seeds_per_prompt: seeds,
            total_generations: 0,
            mean_style_distance: 0.0,
            median_style_distance: 0.0,
            mean_fk_grade: 0.0,
            mean_questions_per_1k: 0.0,
            mean_exclamations_per_1k: 0.0,
            mean_canon_leakage: 0.0,
            raw_mode: raw,
            adapter_used: adapter,
            model: model_id.to_string(),
        };
    }

    let mut distances: Vec<f64> = records.iter().map(|r| r.style_distance).collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if distances.len() % 2 == 0 {
        let mid = distances.len() / 2;
        (distances[mid - 1] + distances[mid]) / 2.0
    } else {
        distances[distances.len() / 2]
    };

    let unique_prompts: std::collections::HashSet<&str> =
        records.iter().map(|r| r.prompt.as_str()).collect();

    EvalSummary {
        suite_name: suite_name.to_string(),
        total_prompts: unique_prompts.len(),
        seeds_per_prompt: seeds,
        total_generations: records.len(),
        mean_style_distance: records.iter().map(|r| r.style_distance).sum::<f64>() / n,
        median_style_distance: median,
        mean_fk_grade: records.iter().map(|r| r.fk_grade).sum::<f64>() / n,
        mean_questions_per_1k: records.iter().map(|r| r.questions_per_1k).sum::<f64>() / n,
        mean_exclamations_per_1k: records.iter().map(|r| r.exclamations_per_1k).sum::<f64>() / n,
        mean_canon_leakage: records.iter().map(|r| r.canon_leakage_score).sum::<f64>() / n,
        raw_mode: raw,
        adapter_used: adapter,
        model: model_id.to_string(),
    }
}
