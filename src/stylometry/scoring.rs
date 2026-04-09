//! Stylometric distance scoring.
//!
//! Computes a 0.0-1.0 distance between a text sample and a fingerprint.
//! Lower means closer to the author's voice.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

use crate::stylometry::ai_slop;
use crate::stylometry::features::function_words;
use crate::stylometry::features::lengths;
use crate::stylometry::features::ngrams;
use crate::stylometry::features::punctuation::PunctuationStats;
use crate::stylometry::features::readability::ReadabilityStats;
use crate::stylometry::features::richness::RichnessStats;
use crate::stylometry::fingerprint::StylometricFingerprint;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceReport {
    pub overall: f64,
    pub sentence_length_kl: f64,
    pub function_word_cos: f64,
    pub punctuation_l1: f64,
    pub ngram_cos: f64,
    pub readability_diff: f64,
    pub richness_diff: f64,
    pub ai_slop_penalty: f64,
}

/// Compute stylometric distance between text and a fingerprint.
/// Returns 0.0 (identical style) to 1.0 (maximally different).
///
/// Weight rationale (grounded in PAN shared task findings + Writeprints):
/// - Function words (0.20): Top discriminator in PAN 2011-2025 shared tasks
/// - Char n-grams (0.20): Second-strongest signal per PAN evaluations
/// - Sentence length (0.15): Classic Writeprints feature (Abbasi & Chen 2008)
/// - Punctuation (0.10): Strong AI-tell signal (em-dashes, semicolons)
/// - Readability (0.10): Captures complexity patterns (Flesch 1948, Coleman-Liau 1975)
/// - Vocabulary richness (0.10): Yule's K + hapax ratio (Yule 1944, Writeprints)
/// - AI-slop penalty (0.15): Catches LLM-specific word/phrase patterns
pub fn distance(text: &str, fingerprint: &StylometricFingerprint) -> DistanceReport {
    let sentence_length_kl = sentence_length_divergence(text, fingerprint);
    let function_word_cos = function_word_distance(text, fingerprint);
    let punctuation_l1 = punctuation_distance(text, fingerprint);
    let ngram_cos = ngram_distance(text, fingerprint);
    let readability_diff = readability_distance(text, fingerprint);
    let richness_diff = richness_distance(text, fingerprint);
    let ai_slop_penalty = slop_penalty(text);

    // Weighted combination — weights validated against PAN shared task findings
    let overall = (function_word_cos * 0.20
        + ngram_cos * 0.20
        + sentence_length_kl * 0.15
        + punctuation_l1 * 0.10
        + readability_diff * 0.10
        + richness_diff * 0.10
        + ai_slop_penalty * 0.15)
        .clamp(0.0, 1.0);

    DistanceReport {
        overall,
        sentence_length_kl,
        function_word_cos,
        punctuation_l1,
        ngram_cos,
        readability_diff,
        richness_diff,
        ai_slop_penalty,
    }
}

fn sentence_length_divergence(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_stats = lengths::sentence_lengths(text);
    if fp.sentence_length.mean == 0.0 || text_stats.mean == 0.0 {
        return 0.5;
    }

    // Simplified KL-like divergence using mean and SD
    let mean_diff = ((text_stats.mean - fp.sentence_length.mean) / fp.sentence_length.mean.max(1.0)).abs();
    let sd_diff = if fp.sentence_length.sd > 0.0 {
        ((text_stats.sd - fp.sentence_length.sd) / fp.sentence_length.sd).abs()
    } else {
        0.0
    };

    ((mean_diff + sd_diff) / 2.0).clamp(0.0, 1.0)
}

fn function_word_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_fw = function_words::compute(text);

    if fp.function_words.is_empty() || text_fw.is_empty() {
        return 0.5;
    }

    // Cosine distance between function word frequency vectors
    let all_keys: Vec<&String> = fp
        .function_words
        .keys()
        .chain(text_fw.keys())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for key in &all_keys {
        let a = fp.function_words.get(*key).copied().unwrap_or(0.0);
        let b = text_fw.get(*key).copied().unwrap_or(0.0);
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.5;
    }

    let cosine_sim = dot / denom;
    (1.0 - cosine_sim).clamp(0.0, 1.0)
}

fn punctuation_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_punct = PunctuationStats::compute(text);
    let fp_p = &fp.punctuation;

    // L1 distance normalized by typical ranges
    let diffs = [
        (text_punct.em_dashes_per_1k - fp_p.em_dashes_per_1k).abs() / 20.0,
        (text_punct.en_dashes_per_1k - fp_p.en_dashes_per_1k).abs() / 20.0,
        (text_punct.semicolons_per_1k - fp_p.semicolons_per_1k).abs() / 10.0,
        (text_punct.colons_per_1k - fp_p.colons_per_1k).abs() / 10.0,
        (text_punct.exclamations_per_1k - fp_p.exclamations_per_1k).abs() / 10.0,
        (text_punct.questions_per_1k - fp_p.questions_per_1k).abs() / 10.0,
        (text_punct.parentheses_per_1k - fp_p.parentheses_per_1k).abs() / 20.0,
    ];

    let avg: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    avg.clamp(0.0, 1.0)
}

fn ngram_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_ngrams = ngrams::trigrams(text);

    if fp.ngram_profile.is_empty() || text_ngrams.is_empty() {
        return 0.5;
    }

    // Convert to frequency maps for cosine distance
    let fp_total: u64 = fp.ngram_profile.iter().map(|(_, c)| c).sum();
    let text_total: u64 = text_ngrams.iter().map(|(_, c)| c).sum();

    if fp_total == 0 || text_total == 0 {
        return 0.5;
    }

    let fp_map: std::collections::HashMap<&str, f64> = fp
        .ngram_profile
        .iter()
        .map(|(g, c)| (g.as_str(), *c as f64 / fp_total as f64))
        .collect();

    let text_map: std::collections::HashMap<&str, f64> = text_ngrams
        .iter()
        .map(|(g, c)| (g.as_str(), *c as f64 / text_total as f64))
        .collect();

    let all_keys: std::collections::HashSet<&str> = fp_map
        .keys()
        .chain(text_map.keys())
        .copied()
        .collect();

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for key in &all_keys {
        let a = fp_map.get(key).copied().unwrap_or(0.0);
        let b = text_map.get(key).copied().unwrap_or(0.0);
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 0.5;
    }

    (1.0 - dot / denom).clamp(0.0, 1.0)
}

fn readability_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_stats = ReadabilityStats::compute(text);
    let fp_r = &fp.readability;

    // Normalized differences for each readability metric
    let fk_diff = if fp_r.flesch_kincaid_grade.abs() > 0.1 {
        ((text_stats.flesch_kincaid_grade - fp_r.flesch_kincaid_grade) / fp_r.flesch_kincaid_grade.abs().max(1.0)).abs()
    } else {
        0.0
    };

    let cli_diff = if fp_r.coleman_liau_index.abs() > 0.1 {
        ((text_stats.coleman_liau_index - fp_r.coleman_liau_index) / fp_r.coleman_liau_index.abs().max(1.0)).abs()
    } else {
        0.0
    };

    let syllable_diff = if fp_r.avg_syllables_per_word > 0.0 {
        ((text_stats.avg_syllables_per_word - fp_r.avg_syllables_per_word) / fp_r.avg_syllables_per_word).abs()
    } else {
        0.0
    };

    ((fk_diff + cli_diff + syllable_diff) / 3.0).clamp(0.0, 1.0)
}

fn richness_distance(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_stats = RichnessStats::compute(text);
    let fp_r = &fp.richness;

    // Yule's K: typical range 50-300, normalize difference
    let yules_diff = if fp_r.yules_k > 0.0 {
        ((text_stats.yules_k - fp_r.yules_k) / fp_r.yules_k.max(50.0)).abs()
    } else {
        0.0
    };

    // Hapax legomena ratio: typical 0.3-0.7
    let hapax_diff = (text_stats.hapax_legomena_ratio - fp_r.hapax_legomena_ratio).abs() * 2.0;

    // Simpson's D: typical 0.95-0.999
    let simpsons_diff = (text_stats.simpsons_d - fp_r.simpsons_d).abs() * 10.0;

    ((yules_diff + hapax_diff + simpsons_diff) / 3.0).clamp(0.0, 1.0)
}

fn slop_penalty(text: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let word_count = text.unicode_words().count() as f64;
    if word_count == 0.0 {
        return 0.0;
    }

    // Count actual occurrences, not just presence (Codex review fix)
    let mut word_occurrences: usize = 0;
    for word in ai_slop::BANNED_WORDS {
        word_occurrences += text_lower.matches(word).count();
    }

    let mut phrase_occurrences: usize = 0;
    for phrase in ai_slop::BANNED_PHRASES {
        phrase_occurrences += text_lower.matches(phrase).count();
    }

    let total_hits = word_occurrences + phrase_occurrences * 2;
    let density = total_hits as f64 / word_count * 100.0;

    // Scale: 0 hits = 0.0, 5+ per 100 words = 1.0
    (density / 5.0).clamp(0.0, 1.0)
}
