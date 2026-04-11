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

    // Boost preferred words (user's distinctive vocabulary)
    for (word, _freq) in &fp.preferred_words {
        bias.insert(word.clone(), config.preferred_word_bias);
    }

    // Boost punctuation tokens that the fingerprint uses but the model underproduces.
    // The ablation showed 0.0 exclamations/1k vs 5.24 target — the model never uses "!".
    // Token-level bias on punctuation marks closes the structural gap.
    if fp.punctuation.exclamations_per_1k > 1.0 {
        bias.insert("!".to_string(), config.preferred_word_bias * 0.5);
    }
    if fp.punctuation.questions_per_1k > 1.0 {
        bias.insert("?".to_string(), config.preferred_word_bias * 0.3);
    }
    // Em-dashes if the author uses them frequently
    if fp.punctuation.em_dashes_per_1k > 2.0 {
        bias.insert("—".to_string(), config.preferred_word_bias * 0.3);
    }

    bias
}
