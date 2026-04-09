use serde::Serialize;

use crate::config;
use crate::error::AppError;
use crate::output::{self, Ctx};

// ── profile show ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProfileInfo {
    name: String,
    active: bool,
    samples_count: usize,
    total_words: usize,
    total_chars: usize,
    avg_sentence_length: f64,
    sentence_length_sd: f64,
    vocab_size: usize,
    trained: bool,
    samples_dir: String,
}

pub fn show(ctx: Ctx) -> Result<(), AppError> {
    let cfg = config::load()?;
    let info = compute_profile_info(&cfg.active_profile, true)?;

    output::print_success_or(ctx, &info, |p| {
        use owo_colors::OwoColorize;
        println!("Profile: {}", p.name.bold());
        println!(
            "  active:            {}",
            if p.active { "yes".green().to_string() } else { "no".dimmed().to_string() }
        );
        println!("  samples:           {}", p.samples_count.to_string().bold());
        println!("  total words:       {}", p.total_words.to_string().bold());
        println!("  total chars:       {}", p.total_chars.to_string().bold());
        println!("  avg sentence len:  {:.1}", p.avg_sentence_length);
        println!("  sentence len SD:   {:.1}", p.sentence_length_sd);
        println!("  vocabulary size:   {}", p.vocab_size.to_string().bold());
        println!(
            "  trained:           {}",
            if p.trained { "yes".green().to_string() } else { "no".dimmed().to_string() }
        );
        println!("  samples dir:       {}", p.samples_dir.dimmed());
    });

    Ok(())
}

// ── profile list ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProfileSummary {
    name: String,
    active: bool,
    samples_count: usize,
    total_words: usize,
    trained: bool,
}

pub fn list(ctx: Ctx) -> Result<(), AppError> {
    let cfg = config::load()?;
    let profiles_dir = config::profiles_dir();

    if !profiles_dir.exists() {
        return Err(AppError::Config(
            "Profiles directory does not exist. Run: writer init".into(),
        ));
    }

    let mut profiles = Vec::new();
    for entry in std::fs::read_dir(&profiles_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let info = compute_profile_info(&name, name == cfg.active_profile)?;
        profiles.push(ProfileSummary {
            name: info.name,
            active: info.active,
            samples_count: info.samples_count,
            total_words: info.total_words,
            trained: info.trained,
        });
    }

    output::print_success_or(ctx, &profiles, |list| {
        use owo_colors::OwoColorize;
        let mut table = comfy_table::Table::new();
        table.set_header(vec!["Name", "Active", "Samples", "Words", "Trained"]);
        for p in list {
            table.add_row(vec![
                p.name.clone(),
                if p.active {
                    "*".green().to_string()
                } else {
                    "".to_string()
                },
                p.samples_count.to_string(),
                p.total_words.to_string(),
                if p.trained {
                    "yes".green().to_string()
                } else {
                    "no".dimmed().to_string()
                },
            ]);
        }
        println!("{table}");
    });

    Ok(())
}

// ── profile use ─────────────────────────────────────────────────────────────

pub fn use_profile(_ctx: Ctx, _name: String) -> Result<(), AppError> {
    Err(AppError::Transient(
        "profile use: write the name to ~/.config/writer/config.toml manually for now. Wiring up in v0.2.".into()
    ))
}

// ── profile new ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct NewProfileResult {
    name: String,
    samples_dir: String,
    status: String,
}

pub fn new_profile(ctx: Ctx, name: String) -> Result<(), AppError> {
    if name.trim().is_empty() || name.contains('/') || name.contains('\\') {
        return Err(AppError::InvalidInput(
            "profile name must be non-empty and cannot contain path separators".into(),
        ));
    }

    let profile_dir = config::profiles_dir().join(&name);
    let samples_dir = profile_dir.join("samples");

    let status = if samples_dir.exists() {
        "already_exists".to_string()
    } else {
        std::fs::create_dir_all(&samples_dir)?;
        "created".to_string()
    };

    let result = NewProfileResult {
        name: name.clone(),
        samples_dir: samples_dir.display().to_string(),
        status,
    };

    output::print_success_or(ctx, &result, |r| {
        use owo_colors::OwoColorize;
        println!("{} Profile '{}' created", "+".green(), r.name.bold());
        println!("  {}", r.samples_dir.dimmed());
    });

    Ok(())
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn compute_profile_info(name: &str, active: bool) -> Result<ProfileInfo, AppError> {
    let profile_dir = config::profiles_dir().join(name);
    let samples_dir = profile_dir.join("samples");

    if !samples_dir.exists() {
        return Err(AppError::Config(format!(
            "Profile '{name}' does not exist. Create with: writer profile new {name}"
        )));
    }

    let mut samples_count = 0usize;
    let mut total_words = 0usize;
    let mut total_chars = 0usize;
    let mut sentence_lengths: Vec<usize> = Vec::new();
    let mut vocab: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in std::fs::read_dir(&samples_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let content = match std::fs::read_to_string(entry.path()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        samples_count += 1;

        total_chars += content.chars().count();
        let words: Vec<&str> = content.split_whitespace().collect();
        total_words += words.len();

        for w in &words {
            let normalized: String = w
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>()
                .to_lowercase();
            if !normalized.is_empty() {
                vocab.insert(normalized);
            }
        }

        for sentence in content.split(['.', '!', '?']) {
            let len = sentence.split_whitespace().count();
            if len > 0 {
                sentence_lengths.push(len);
            }
        }
    }

    let (avg_sentence_length, sentence_length_sd) = stats(&sentence_lengths);

    let trained = profile_dir.join("adapter.safetensors").exists()
        || profile_dir.join("adapter").exists();

    Ok(ProfileInfo {
        name: name.to_string(),
        active,
        samples_count,
        total_words,
        total_chars,
        avg_sentence_length,
        sentence_length_sd,
        vocab_size: vocab.len(),
        trained,
        samples_dir: samples_dir.display().to_string(),
    })
}

fn stats(values: &[usize]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, variance.sqrt())
}
