//! SlopScore benchmark — AI pattern density detection.
//!
//! Weight: 0.30 of combined score.
//! Uses the stylometry::ai_slop module for word/phrase detection.
//! Falls back to internal detection if the humanise-text Python detector
//! is not available.
use std::path::Path;

use crate::stylometry::ai_slop;

pub struct SlopScoreResult {
    pub ai_density_score: f64,
    pub flagged_words: usize,
    pub flagged_phrases: usize,
    pub method: String,
}

/// Evaluate AI-slop density in generated text.
/// Tries the humanise-text Python detector first, falls back to internal.
pub fn evaluate(text: &str) -> SlopScoreResult {
    // Try Python detector if available
    let python_detector = Path::new(
        &shellexpand::tilde("~/.claude/skills/humanise-text/scripts/detect_ai_patterns.py")
            .to_string(),
    )
    .to_path_buf();

    if python_detector.exists() {
        if let Some(result) = try_python_detector(text, &python_detector) {
            return result;
        }
    }

    // Fallback: internal detector
    evaluate_internal(text)
}

fn try_python_detector(_text: &str, _script_path: &Path) -> Option<SlopScoreResult> {
    // TODO: Wire Python detector once JSON output format is confirmed.
    // For now, fall back to internal detector.
    None
}

fn evaluate_internal(text: &str) -> SlopScoreResult {
    let text_lower = text.to_lowercase();
    let word_count = text.split_whitespace().count() as f64;

    if word_count == 0.0 {
        return SlopScoreResult {
            ai_density_score: 0.0,
            flagged_words: 0,
            flagged_phrases: 0,
            method: "internal".into(),
        };
    }

    let flagged_words: usize = ai_slop::BANNED_WORDS
        .iter()
        .filter(|w| text_lower.contains(**w))
        .count();

    let flagged_phrases: usize = ai_slop::BANNED_PHRASES
        .iter()
        .filter(|p| text_lower.contains(**p))
        .count();

    // AI density: flagged items per 100 words, scaled to 0-100
    let total_flags = flagged_words + flagged_phrases * 2;
    let density = (total_flags as f64 / word_count * 100.0).min(100.0);

    SlopScoreResult {
        ai_density_score: density,
        flagged_words,
        flagged_phrases,
        method: "internal".into(),
    }
}
