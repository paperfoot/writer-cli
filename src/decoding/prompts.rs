//! Prompt templates with stylometric priming.
//!
//! Inject a small fingerprint summary into the system prompt so the model
//! has a concrete target for sentence rhythm and vocabulary.
use crate::stylometry::fingerprint::StylometricFingerprint;

/// Build a system prompt that primes the model with the user's stylometric profile.
///
/// The prompt encodes measurable voice features from the fingerprint so the model
/// has a concrete target for rhythm, punctuation, and vocabulary.
pub fn system_prompt(fingerprint: &StylometricFingerprint) -> String {
    let avg_word_len = fingerprint.word_length.mean;
    let avg_sent_len = fingerprint.sentence_length.mean;
    let sent_sd = fingerprint.sentence_length.sd;

    let preferred: Vec<&str> = fingerprint
        .preferred_words
        .iter()
        .take(8)
        .map(|(w, _)| w.as_str())
        .collect();
    let preferred_str = if preferred.is_empty() {
        "none identified".to_string()
    } else {
        preferred.join(", ")
    };

    // Build punctuation guidance from actual fingerprint rates
    let mut punct_notes = Vec::new();
    if fingerprint.punctuation.questions_per_1k > 2.0 {
        punct_notes.push(format!(
            "Ask rhetorical questions (~{:.0} per 1000 words)",
            fingerprint.punctuation.questions_per_1k
        ));
    }
    if fingerprint.punctuation.exclamations_per_1k > 1.0 {
        punct_notes.push(format!(
            "Use exclamations for emphasis (~{:.0} per 1000 words)",
            fingerprint.punctuation.exclamations_per_1k
        ));
    }
    if fingerprint.punctuation.em_dashes_per_1k > 2.0 {
        punct_notes.push("Use em-dashes for asides and interruptions".to_string());
    }

    let punct_section = if punct_notes.is_empty() {
        String::new()
    } else {
        format!(
            "\n{}",
            punct_notes
                .iter()
                .map(|n| format!("- {n}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    format!(
        "You are a writer with a distinctive voice. Your writing tends to:\n\
         - Mix very short sentences ({:.0}-word fragments) with longer flowing ones (SD {:.1})\n\
         - Favor simple, concrete words (average {avg_word_len:.1} chars per word)\n\
         - Average sentence length around {avg_sent_len:.0} words, but vary widely\
         {punct_section}\n\
         - Include dry wit, understatement, and comic timing\n\
         - Words you favor: {preferred_str}\n\
         \n\
         Avoid these AI-sounding words and phrases: delve, tapestry, landscape, \
         leverage, nuance, multifaceted, holistic, pivotal, \
         it's worth noting, in today's world, the intersection of.\n\
         \n\
         Write naturally. Output only the requested text.",
        (avg_sent_len - sent_sd).max(1.0),
        sent_sd,
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
