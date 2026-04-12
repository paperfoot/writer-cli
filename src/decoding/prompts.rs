//! Prompt templates with stylometric priming.
//!
//! Register-style descriptors per GPT Pro review (Prompt-and-Rerank, arXiv:2205.11503).
//! Natural-language style descriptors outperform metric sheets.
use crate::stylometry::fingerprint::StylometricFingerprint;

/// Build a system prompt that primes the model with the user's stylometric profile.
pub fn system_prompt(fingerprint: &StylometricFingerprint) -> String {
    let avg_sent_len = fingerprint.sentence_length.mean;
    let sent_sd = fingerprint.sentence_length.sd;

    let preferred: Vec<&str> = fingerprint
        .preferred_words
        .iter()
        .take(8)
        .map(|(w, _)| w.as_str())
        .collect();
    let preferred_str = preferred.join(", ");

    let mut notes = vec![
        "Match the author's rhythm and register, not the training corpus subject matter."
            .to_string(),
        format!(
            "Alternate clipped sentences ({:.0}-word fragments) with longer flowing ones.",
            (avg_sent_len - sent_sd).max(1.0)
        ),
        "Prefer concrete, everyday diction over abstract jargon.".to_string(),
    ];

    if fingerprint.punctuation.em_dashes_per_1k > 2.0 {
        notes.push("Use occasional em-dash asides.".to_string());
    }

    notes.push(
        "Use asides sparingly; punctuation should feel incidental, not performative.".to_string(),
    );

    if !preferred_str.is_empty() {
        notes.push(format!("Words you naturally reach for: {preferred_str}."));
    }

    notes.push(
        "Avoid these AI-sounding words: delve, tapestry, landscape, leverage, nuance, \
         multifaceted, holistic, pivotal."
            .to_string(),
    );
    notes.push(
        "Do not reuse recurring names or universe-specific nouns unless the user asks for them."
            .to_string(),
    );
    notes.push("Keep the answer on-topic. Output only the requested text.".to_string());

    notes.join("\n")
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
