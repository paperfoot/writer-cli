//! Prompt relevance scoring.
//!
//! Measures whether generated text addresses the prompt's topic.
//! Without this, the ranker could prefer a beautifully-styled off-topic
//! response over a relevant one.
//!
//! Approach: content-word overlap — extract meaningful words from the prompt
//! (excluding stop words), then measure what fraction appear in the output.
//! Simple, fast, and sufficient for a generation-time gate.

use unicode_segmentation::UnicodeSegmentation;

/// Common English stop words — excluded from content-word extraction.
/// Kept minimal to avoid false negatives on short prompts.
const STOP_WORDS: &[&str] = &[
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "our",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "when",
    "where",
    "why",
    "not",
    "no",
    "so",
    "if",
    "about",
    "up",
    "out",
    "just",
    "than",
    "then",
    "also",
    "very",
    "some",
    "any",
    "all",
    "each",
    "every",
    "into",
    "as",
    "write",
    "writing",
    "paragraph",
    "essay",
    "piece",
    "about",
    "describe",
    "explain",
    "tell",
];

/// Extract content words from text — lowercase, alphabetic, non-stop-word.
fn content_words(text: &str) -> Vec<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 3 && w.chars().all(|c| c.is_alphabetic()))
        .filter(|w| !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

/// Compute prompt relevance as the fraction of prompt content words
/// that appear at least once in the output.
///
/// Returns a score in [0.0, 1.0] where:
/// - 1.0 = every prompt content word appears in the output
/// - 0.0 = no prompt content words appear in the output
///
/// If the prompt has no content words (e.g., "Write something"), returns 1.0
/// to avoid penalizing vague prompts.
pub fn score(prompt: &str, output: &str) -> f64 {
    let prompt_words = content_words(prompt);
    if prompt_words.is_empty() {
        return 1.0;
    }

    // Deduplicate prompt words
    let unique_prompt: std::collections::HashSet<&str> =
        prompt_words.iter().map(|s| s.as_str()).collect();

    let _output_lower = output.to_lowercase();
    let output_words: std::collections::HashSet<String> =
        output.unicode_words().map(|w| w.to_lowercase()).collect();

    let mut found = 0;
    for word in &unique_prompt {
        if output_words.contains(*word) {
            found += 1;
        } else {
            // Fuzzy: check shared prefix >= 4 chars for morphological variants.
            // "dolphins" matches "dolphin", "swimming" matches "swims", etc.
            let min_prefix = word.len().min(4);
            let prefix = &word[..min_prefix];
            if output_words
                .iter()
                .any(|ow| ow.starts_with(prefix) && common_prefix_len(word, ow) >= min_prefix)
            {
                found += 1;
            }
        }
    }

    found as f64 / unique_prompt.len() as f64
}

/// Length of the common prefix between two strings.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_relevance() {
        let prompt = "Write about dolphins and ocean life";
        let output =
            "Dolphins are fascinating creatures of the ocean. Their life underwater is complex.";
        let s = score(prompt, output);
        assert!(s >= 0.5, "score {s} should be at least 0.5");
    }

    #[test]
    fn zero_relevance() {
        let prompt = "Write about quantum physics and black holes";
        let output = "The garden was beautiful with roses and tulips blooming everywhere.";
        let s = score(prompt, output);
        assert!(s < 0.3, "score {s} should be low for off-topic output");
    }

    #[test]
    fn vague_prompt_returns_one() {
        // All words are stop words or too short → no content words → 1.0
        let prompt = "Write about it for me";
        let output = "Anything at all.";
        assert_eq!(score(prompt, output), 1.0);
    }

    #[test]
    fn empty_output() {
        let prompt = "Write about dolphins";
        assert_eq!(score(prompt, ""), 0.0);
    }

    #[test]
    fn morphological_variant() {
        let prompt = "Write about dolphins swimming";
        let output = "A dolphin swims gracefully through the water.";
        let s = score(prompt, output);
        // "dolphins" should partially match "dolphin" via substring
        assert!(s > 0.0, "score {s} should be > 0 for morphological matches");
    }
}
