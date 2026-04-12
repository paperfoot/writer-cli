//! Logit bias computation from stylometric fingerprint.
//!
//! Suppress AI-tell words and boost user's preferred vocabulary.
//! Reference: Writeprints (Abbasi & Chen, 2008) — vocabulary-level bias.
use std::collections::HashMap;

use crate::backends::inference::request::LogitBiasMap;
use crate::config::DecodingConfig;
use crate::stylometry::ai_slop;
use crate::stylometry::fingerprint::StylometricFingerprint;

/// Build a logit bias map from the user's fingerprint.
/// Banned words get negative bias, preferred words get positive bias.
pub fn from_fingerprint(fp: &StylometricFingerprint, config: &DecodingConfig) -> LogitBiasMap {
    let mut bias: HashMap<String, f32> = HashMap::new();

    // Suppress banned words (AI slop not in user's vocabulary)
    for word in &fp.banned_words {
        bias.insert(word.clone(), config.banned_word_bias);
    }

    // Also suppress the static AI slop list, excluding author-valid discourse markers
    const AUTHOR_VALID: &[&str] = &["furthermore", "moreover", "nevertheless"];
    for word in ai_slop::BANNED_WORDS {
        if AUTHOR_VALID.contains(word) {
            continue;
        }
        bias.entry(word.to_string())
            .or_insert(config.banned_word_bias);
    }

    // Positive boosting removed per GPT Pro review (arXiv:2205.11503, Stamatatos survey):
    // preferred_words contains canon-heavy content words (arthur, zaphod, universe)
    // that help leakage more than style. Punctuation boosts were overshooting
    // targets (17.2 Q/1k vs 8.95 target, 10.6 !/1k vs 5.24 target).
    // Only em-dash gets a small boost — it's structural, not content.
    if fp.punctuation.em_dashes_per_1k > 2.0 {
        bias.insert("—".to_string(), config.preferred_word_bias * 0.2);
    }

    bias
}
