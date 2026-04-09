//! Prompt templates with stylometric priming.
//!
//! Inject a small fingerprint summary into the system prompt so the model
//! has a concrete target for sentence rhythm and vocabulary.
use crate::stylometry::fingerprint::StylometricFingerprint;

/// Build a system prompt that primes the model with the user's stylometric profile.
pub fn system_prompt(fingerprint: &StylometricFingerprint) -> String {
    let avg_sentence_len = fingerprint.sentence_length.mean;
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
    let em_dashes = fingerprint.punctuation.em_dashes_per_1k;
    let semicolons = fingerprint.punctuation.semicolons_per_1k;

    format!(
        "Write in the following style profile:\n\
         - Average sentence length: {avg_sentence_len:.0} words\n\
         - Average word length: {avg_word_len:.1} characters\n\
         - Distinctive vocabulary: {preferred_str}\n\
         - Em-dashes per 1000 words: {em_dashes:.1}\n\
         - Semicolons per 1000 words: {semicolons:.1}\n\
         - Readability: grade level {grade:.0}\n\
         \n\
         Match this rhythm and vocabulary. Do not use AI-sounding transitions \
         like 'Moreover', 'Furthermore', 'It's worth noting'. \
         Do not use em-dashes unless the profile shows them. \
         Output only the requested text, nothing else.",
        grade = fingerprint.readability.flesch_kincaid_grade
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
