//! Prompt templates with stylometric priming.
//!
//! Inject a small fingerprint summary into the system prompt so the model
//! has a concrete target for sentence rhythm and vocabulary.
use crate::stylometry::fingerprint::StylometricFingerprint;

/// Build a system prompt that primes the model with the user's stylometric profile.
pub fn system_prompt(fingerprint: &StylometricFingerprint) -> String {
    let _avg_sentence_len = fingerprint.sentence_length.mean;
    let avg_word_len = fingerprint.word_length.mean;

    // Top 5 preferred words (distinctive vocabulary)
    let preferred: Vec<&str> = fingerprint
        .preferred_words
        .iter()
        .take(5)
        .map(|(w, _)| w.as_str())
        .collect();
    let preferred_str = if preferred.is_empty() {
        "none identified".to_string()
    } else {
        preferred.join(", ")
    };

    // Punctuation profile summary
    let _em_dashes = fingerprint.punctuation.em_dashes_per_1k;
    let _semicolons = fingerprint.punctuation.semicolons_per_1k;

    format!(
        "You are a writer with a distinctive voice. Your writing tends to:\n\
         - Use short, punchy sentences mixed with occasional longer ones\n\
         - Favor simple, concrete words (average {avg_word_len:.1} chars per word)\n\
         - Use questions and exclamations freely\n\
         - Include dry wit and understatement\n\
         - Words you favor: {preferred_str}\n\
         \n\
         Avoid these AI-sounding words and phrases: delve, tapestry, landscape, \
         leverage, nuance, multifaceted, holistic, pivotal, moreover, furthermore, \
         it's worth noting, in today's world, the intersection of.\n\
         \n\
         Write naturally. Output only the requested text.",
    )
}

/// Build a write prompt.
pub fn write_prompt(user_input: &str) -> String {
    user_input.to_string()
}

/// Build a rewrite prompt.
pub fn rewrite_prompt(original: &str) -> String {
    format!(
        "Rewrite the following text while preserving its meaning and the author's voice. \
         Keep the same level of formality, vocabulary preferences, and sentence rhythm. \
         Do not add flourishes or structure the original did not have. \
         Output only the rewritten text.\n\n\
         ORIGINAL:\n{original}\n\nREWRITTEN:"
    )
}
