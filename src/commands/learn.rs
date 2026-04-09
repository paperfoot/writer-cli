use serde::Serialize;
use std::collections::HashSet;
use std::path::PathBuf;

use writer_cli::corpus::ingest;
use writer_cli::corpus::sample::Sample;
use writer_cli::stylometry::fingerprint::StylometricFingerprint;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct LearnResult {
    profile: String,
    samples_added: usize,
    samples_skipped_dedupe: usize,
    total_words: usize,
    fingerprint_word_count: u64,
    fingerprint_sentence_length_mean: f64,
    fingerprint_vocabulary_size: u64,
    samples_dir: String,
}

pub fn run(ctx: Ctx, files: Vec<PathBuf>) -> Result<(), AppError> {
    let cfg = config::load()?;
    let profile_dir = config::profiles_dir().join(&cfg.active_profile);
    let samples_dir = profile_dir.join("samples");

    if !profile_dir.exists() {
        return Err(AppError::Config(format!(
            "Profile '{}' does not exist. Run: writer init",
            cfg.active_profile
        )));
    }

    std::fs::create_dir_all(&samples_dir)?;

    // Load existing sample hashes for dedup
    let existing_hashes = load_existing_hashes(&samples_dir);

    // Run the real ingest pipeline
    let (samples, report) = ingest::ingest(
        &files,
        None, // TODO: --context flag
        cfg.training.max_seq_len,
        &existing_hashes,
        true, // normalize by default
    )
    .map_err(|e| AppError::Transient(e.to_string()))?;

    // Write samples as JSONL
    let jsonl_path = samples_dir.join("corpus.jsonl");
    let mut jsonl_content = if jsonl_path.exists() {
        std::fs::read_to_string(&jsonl_path).unwrap_or_default()
    } else {
        String::new()
    };

    for sample in &samples {
        if let Ok(line) = serde_json::to_string(sample) {
            jsonl_content.push_str(&line);
            jsonl_content.push('\n');
        }
    }
    std::fs::write(&jsonl_path, &jsonl_content)?;

    // Compute fingerprint on all samples (existing + new)
    let all_samples = load_all_samples(&samples_dir);
    let fingerprint = StylometricFingerprint::compute(&all_samples);

    // Save fingerprint
    let fp_path = profile_dir.join("fingerprint.json");
    let fp_json = serde_json::to_string_pretty(&fingerprint)
        .map_err(|e| AppError::Transient(e.to_string()))?;
    std::fs::write(&fp_path, &fp_json)?;

    let result = LearnResult {
        profile: cfg.active_profile.clone(),
        samples_added: report.samples_added,
        samples_skipped_dedupe: report.samples_skipped_dedupe,
        total_words: report.total_words,
        fingerprint_word_count: fingerprint.word_count,
        fingerprint_sentence_length_mean: fingerprint.sentence_length.mean,
        fingerprint_vocabulary_size: fingerprint.vocabulary_size,
        samples_dir: samples_dir.display().to_string(),
    };

    output::print_success_or(ctx, &result, |r| {
        use owo_colors::OwoColorize;
        println!(
            "{} Ingested {} samples into profile '{}'",
            "+".green(),
            r.samples_added.to_string().bold(),
            r.profile.bold()
        );
        if r.samples_skipped_dedupe > 0 {
            println!(
                "  {} {} duplicates skipped",
                "!".yellow(),
                r.samples_skipped_dedupe
            );
        }
        println!("  {} total words", r.total_words.to_string().bold());
        println!(
            "  fingerprint: {} words, {:.1} avg sentence length, {} vocabulary",
            r.fingerprint_word_count,
            r.fingerprint_sentence_length_mean,
            r.fingerprint_vocabulary_size
        );
        println!();
        println!("Next: {}", "writer train".bold());
    });

    Ok(())
}

fn load_existing_hashes(samples_dir: &std::path::Path) -> HashSet<String> {
    let mut hashes = HashSet::new();
    let jsonl_path = samples_dir.join("corpus.jsonl");
    if let Ok(content) = std::fs::read_to_string(jsonl_path) {
        for line in content.lines() {
            if let Ok(sample) = serde_json::from_str::<Sample>(line) {
                hashes.insert(sample.content_hash);
            }
        }
    }
    hashes
}

fn load_all_samples(samples_dir: &std::path::Path) -> Vec<Sample> {
    let mut samples = Vec::new();
    let jsonl_path = samples_dir.join("corpus.jsonl");
    if let Ok(content) = std::fs::read_to_string(jsonl_path) {
        for line in content.lines() {
            if let Ok(sample) = serde_json::from_str::<Sample>(line) {
                samples.push(sample);
            }
        }
    }
    samples
}
