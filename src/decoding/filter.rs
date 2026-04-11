//! Post-hoc structural filtering of generated text.
//!
//! Hard-reject outputs that contain banned words or phrases,
//! or that fail structural quality checks.
use crate::config::DecodingConfig;
use crate::stylometry::ai_slop;
use crate::stylometry::fingerprint::StylometricFingerprint;

pub struct FilterResult {
    pub passed: bool,
    pub reasons: Vec<String>,
}

/// Detect repetition collapse by checking for repeated n-grams.
/// Returns 0.0 (no repetition) to 1.0 (fully repeated).
fn detect_repetition(text: &str) -> f64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 20 {
        return 0.0;
    }

    // Check 4-gram repetition rate
    let mut seen = std::collections::HashMap::new();
    let mut repeated_positions = 0usize;

    for window in words.windows(4) {
        let key = window.join(" ");
        let count = seen.entry(key).or_insert(0usize);
        *count += 1;
        if *count > 1 {
            repeated_positions += 1;
        }
    }

    let total_windows = words.len().saturating_sub(3);
    if total_windows == 0 {
        return 0.0;
    }

    repeated_positions as f64 / total_windows as f64
}

/// Check a generated text against quality criteria.
pub fn check(
    text: &str,
    _fingerprint: &StylometricFingerprint,
    _config: &DecodingConfig,
) -> FilterResult {
    let text_lower = text.to_lowercase();
    let mut reasons = Vec::new();

    // Check for banned phrases (hard reject)
    for phrase in ai_slop::BANNED_PHRASES {
        if text_lower.contains(phrase) {
            reasons.push(format!("banned phrase: \"{phrase}\""));
        }
    }

    // Check for high concentration of banned words (soft reject if > 3)
    // Exclude author-valid discourse markers that appear in the target corpus
    const AUTHOR_VALID: &[&str] = &["furthermore", "moreover", "nevertheless"];
    let banned_count: usize = ai_slop::BANNED_WORDS
        .iter()
        .filter(|w| !AUTHOR_VALID.contains(*w) && text_lower.contains(**w))
        .count();

    if banned_count > 3 {
        reasons.push(format!("{banned_count} banned words detected"));
    }

    // Check for empty or trivially short output
    let word_count = text.split_whitespace().count();
    if word_count < 10 {
        reasons.push("output too short (< 10 words)".to_string());
    }

    // Check for repetition collapse — catch outputs where the model loops
    if word_count > 20 {
        let repetition_ratio = detect_repetition(text);
        if repetition_ratio > 0.4 {
            reasons.push(format!(
                "repetition collapse detected ({:.0}% repeated phrases)",
                repetition_ratio * 100.0
            ));
        }
    }

    FilterResult {
        passed: reasons.is_empty(),
        reasons,
    }
}
