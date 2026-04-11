//! Stylometric fingerprint computation.
//!
//! References:
//! - Abbasi, A. & Chen, H. (2008). "Writeprints: A stylometric approach to
//!   identity-level identification and similarity detection in cyberspace."
//!   ACM TOIS 26(2). — Foundational feature taxonomy.
//! - Rivera-Soto et al. (EMNLP 2021). "LUAR: Linguistic Understanding and
//!   Attribution of Relations." — Neural authorship embeddings.
//! - Yule, G.U. (1944). "The Statistical Study of Literary Vocabulary." — Yule's K.
//! - PAN Shared Tasks (2011-2025). — Validated function words + char n-grams
//!   as top discriminative signals for authorship verification.
//! - "Layered Insights" (EMNLP 2025). — Confirms hybrid (interpretable +
//!   neural) outperforms either alone.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::corpus::sample::Sample;
use crate::stylometry::features::function_words;
use crate::stylometry::features::lengths::{self, LengthStats};
use crate::stylometry::features::ngrams;
use crate::stylometry::features::punctuation::PunctuationStats;
use crate::stylometry::features::readability::ReadabilityStats;
use crate::stylometry::features::richness::RichnessStats;
use crate::stylometry::features::vocabulary;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StylometricFingerprint {
    pub word_count: u64,
    pub char_count: u64,
    pub word_length: LengthStats,
    pub sentence_length: LengthStats,
    pub paragraph_length: LengthStats,
    pub function_words: HashMap<String, f64>,
    pub ngram_profile: Vec<(String, u64)>,
    pub punctuation: PunctuationStats,
    pub readability: ReadabilityStats,
    pub richness: RichnessStats,
    pub vocabulary_size: u64,
    pub banned_words: Vec<String>,
    pub preferred_words: Vec<(String, f64)>,
}

impl StylometricFingerprint {
    /// Compute a fingerprint from a collection of writing samples.
    pub fn compute(samples: &[Sample]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let full_text: String = samples
            .iter()
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let word_count = samples.iter().map(|s| s.word_count() as u64).sum();
        let char_count = samples.iter().map(|s| s.char_count() as u64).sum();

        let word_length = lengths::word_lengths(&full_text);
        let mut sentence_length = lengths::sentence_lengths(&full_text);
        sentence_length.histogram = lengths::sentence_length_histogram(&full_text);
        let paragraph_length = lengths::paragraph_lengths(&full_text);
        let function_words_freq = function_words::compute(&full_text);
        let trigram_profile = ngrams::trigrams(&full_text);
        let punctuation = PunctuationStats::compute(&full_text);
        let readability = ReadabilityStats::compute(&full_text);
        let richness = RichnessStats::compute(&full_text);
        let vocab_analysis = vocabulary::analyze(&full_text);

        Self {
            word_count,
            char_count,
            word_length,
            sentence_length,
            paragraph_length,
            function_words: function_words_freq,
            ngram_profile: trigram_profile,
            punctuation,
            readability,
            richness,
            vocabulary_size: vocab_analysis.vocabulary_size,
            banned_words: vocab_analysis.banned_words,
            preferred_words: vocab_analysis.preferred_words,
        }
    }
}
