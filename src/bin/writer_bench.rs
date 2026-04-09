//! writer-bench — benchmark runner for the autoresearch loop.
//!
//! Usage:
//!   writer-bench run [--json] [--smoke] [--component voice|slop|creative]
//!   writer-bench baseline --save
//!   writer-bench compare --against baseline
use std::collections::HashSet;
use std::path::PathBuf;

use writer_cli::bench::combined;
use writer_cli::bench::slop_score;
use writer_cli::bench::voice_fidelity;
use writer_cli::corpus::ingest;
use writer_cli::corpus::sample::Sample;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_output = args.iter().any(|a| a == "--json");
    let smoke = args.iter().any(|a| a == "--smoke");

    let subcommand = args.get(1).map(|s| s.as_str()).unwrap_or("run");

    match subcommand {
        "run" => run_benchmarks(json_output, smoke),
        "baseline" => {
            let save = args.iter().any(|a| a == "--save");
            run_baseline(save, json_output);
        }
        _ => {
            eprintln!("Usage: writer-bench run [--json] [--smoke]");
            eprintln!("       writer-bench baseline --save");
            std::process::exit(3);
        }
    }
}

fn run_benchmarks(json_output: bool, smoke: bool) {
    // Load user samples from default profile
    let profile_dir = writer_cli_data_dir().join("profiles/default/samples");
    let samples = load_samples(&profile_dir);

    if samples.is_empty() {
        eprintln!("No samples found in profile. Run: writer learn <files>");
        std::process::exit(2);
    }

    // Split into training and held-out
    let (train_samples, holdout_samples) = voice_fidelity::holdout_split(&samples, 0.1);

    // Compute fingerprint from training set
    let fingerprint = StylometricFingerprint::compute(&train_samples);

    // For smoke test, use held-out samples as "generated" texts
    // (this tests the scoring pipeline, not actual generation)
    let eval_texts: Vec<String> = if smoke {
        holdout_samples
            .iter()
            .take(3)
            .map(|s| s.content.clone())
            .collect()
    } else {
        // In full mode, we'd generate via Ollama. For now, use held-out.
        holdout_samples.iter().map(|s| s.content.clone()).collect()
    };

    // Benchmark 1: VoiceFidelity
    let voice_result = voice_fidelity::evaluate(&eval_texts, &fingerprint);

    // Benchmark 2: SlopScore (average across generated texts)
    let slop_results: Vec<_> = eval_texts.iter().map(|t| slop_score::evaluate(t)).collect();
    let avg_slop = if slop_results.is_empty() {
        slop_score::SlopScoreResult {
            ai_density_score: 0.0,
            flagged_words: 0,
            flagged_phrases: 0,
            method: "none".into(),
        }
    } else {
        let avg_density =
            slop_results.iter().map(|r| r.ai_density_score).sum::<f64>() / slop_results.len() as f64;
        let total_words: usize = slop_results.iter().map(|r| r.flagged_words).sum();
        let total_phrases: usize = slop_results.iter().map(|r| r.flagged_phrases).sum();
        slop_score::SlopScoreResult {
            ai_density_score: avg_density,
            flagged_words: total_words,
            flagged_phrases: total_phrases,
            method: slop_results[0].method.clone(),
        }
    };

    // Benchmark 3: CreativeWritingRubric (placeholder score until EQ-Bench is wired)
    let creative_score = 72.0; // Baseline placeholder

    // Combined score
    let combined = combined::compute(&voice_result, &avg_slop, creative_score);

    if json_output {
        let output = serde_json::json!({
            "version": "1",
            "status": "success",
            "data": {
                "combined_score": combined.combined_score,
                "voice_fidelity": {
                    "distance": combined.voice_fidelity_distance,
                    "component": combined.voice_fidelity_component,
                    "n_samples": voice_result.n_samples,
                },
                "slop_score": {
                    "density": combined.slop_density,
                    "component": combined.slop_component,
                    "flagged_words": avg_slop.flagged_words,
                    "flagged_phrases": avg_slop.flagged_phrases,
                    "method": avg_slop.method,
                },
                "creative_rubric": {
                    "score": combined.creative_rubric_score,
                    "component": combined.creative_rubric_component,
                },
                "smoke": smoke,
            }
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        println!("writer-bench results");
        println!("====================");
        println!();
        println!(
            "Combined score:      {:.3}",
            combined.combined_score
        );
        println!();
        println!(
            "VoiceFidelity:       {:.3} distance ({} samples)",
            combined.voice_fidelity_distance, voice_result.n_samples
        );
        println!(
            "  component:         {:.3} (weight 0.50)",
            combined.voice_fidelity_component
        );
        println!();
        println!(
            "SlopScore:           {:.1} density ({} words, {} phrases flagged)",
            combined.slop_density, avg_slop.flagged_words, avg_slop.flagged_phrases
        );
        println!(
            "  component:         {:.3} (weight 0.30)",
            combined.slop_component
        );
        println!();
        println!(
            "CreativeRubric:      {:.0}/100",
            combined.creative_rubric_score
        );
        println!(
            "  component:         {:.3} (weight 0.20)",
            combined.creative_rubric_component
        );
        if smoke {
            println!();
            println!("(smoke test — used held-out samples, not generated text)");
        }
    }
}

fn run_baseline(save: bool, json_output: bool) {
    // Run benchmarks and optionally save as baseline
    run_benchmarks(json_output, false);

    if save {
        eprintln!("Baseline saving not yet implemented — capture the JSON output manually.");
    }
}

fn load_samples(dir: &std::path::Path) -> Vec<Sample> {
    if !dir.exists() {
        return Vec::new();
    }

    // Try to load JSONL files from the samples dir
    let mut samples = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "jsonl" || e == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    for line in content.lines() {
                        if let Ok(sample) = serde_json::from_str::<Sample>(line) {
                            samples.push(sample);
                        }
                    }
                }
            }
        }
    }

    // If no JSONL, try ingesting md files directly
    if samples.is_empty() {
        if let Ok((ingested, _)) = ingest::ingest(
            &[dir.to_path_buf()],
            None,
            4096,
            &HashSet::new(),
            true,
        ) {
            samples = ingested;
        }
    }

    samples
}

fn writer_cli_data_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "writer")
        .map(|d| d.data_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}
