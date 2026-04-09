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
    let banned_count: usize = ai_slop::BANNED_WORDS
        .iter()
        .filter(|w| text_lower.contains(**w))
        .count();

    if banned_count > 3 {
        reasons.push(format!("{banned_count} banned words detected"));
    }

    // Check for empty or trivially short output
    let word_count = text.split_whitespace().count();
    if word_count < 10 {
        reasons.push("output too short (< 10 words)".to_string());
    }

    FilterResult {
        passed: reasons.is_empty(),
        reasons,
    }
}
