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
    /// Voice distance before slop penalty
    pub base_voice_distance: f64,
    pub sentence_length_dist: f64,
    pub function_word_cos: f64,
    /// Terminal punctuation: questions + exclamations
    pub terminal_punct_dist: f64,
    /// Structural punctuation: em-dashes, semicolons, colons, parens
    pub structural_punct_dist: f64,
    pub ngram_cos: f64,
    pub readability_diff: f64,
    pub richness_diff: f64,
    /// Raw slop score (0.0 = clean, higher = more slop)
    pub slop_score: f64,
    /// Multiplier applied to base distance (1.0 = no penalty)
    pub slop_multiplier: f64,
}

/// Compute stylometric distance between text and a fingerprint.
/// Returns 0.0 (identical style) to 1.0 (maximally different).
///
/// Weight rationale (grounded in PAN shared task findings + Writeprints):
/// - Function words (0.25): Top discriminator in PAN 2011-2025 shared tasks
/// - Char n-grams (0.20): Second-strongest signal per PAN evaluations
/// - Sentence length (0.15): Classic Writeprints feature (Abbasi & Chen 2008)
/// - Terminal punctuation (0.10): Questions + exclamations — biggest gap for Adams
/// - Structural punctuation (0.10): Em-dashes, semicolons, colons, parentheses
/// - Readability (0.10): Captures complexity patterns (Flesch 1948, Coleman-Liau 1975)
/// - Vocabulary richness (0.10): Yule's K + hapax ratio (Yule 1944, Writeprints)
///
/// AI-slop is a post-score penalty multiplier, not a weighted component.
/// It uses a dead-zone + quadratic ramp so small amounts of slop are tolerated
/// but heavy slop aggressively penalizes the score. (GPT Pro round 3 recommendation)
pub fn distance(text: &str, fingerprint: &StylometricFingerprint) -> DistanceReport {
    let sentence_length_dist = sentence_length_divergence(text, fingerprint);
    let function_word_cos = function_word_distance(text, fingerprint);
    let (terminal_punct_dist, structural_punct_dist) = punctuation_distance_split(text, fingerprint);
    let ngram_cos = ngram_distance(text, fingerprint);
    let readability_diff = readability_distance(text, fingerprint);
    let richness_diff = richness_distance(text, fingerprint);
    let slop_score = compute_slop_score(text);

    // Weighted voice distance (without slop)
    let base_voice_distance = (function_word_cos * 0.25
        + ngram_cos * 0.20
        + sentence_length_dist * 0.15
        + terminal_punct_dist * 0.10
        + structural_punct_dist * 0.10
        + readability_diff * 0.10
        + richness_diff * 0.10)
        .clamp(0.0, 1.0);

    // AI-slop: dead-zone + quadratic penalty multiplier
    // Dead zone: slop < 0.02 → no penalty (tolerates detector noise)
    // Ramp: quadratic from 0.02 to 0.10, then saturates
    let x = ((slop_score - 0.02).max(0.0) / 0.08).min(1.0);
    let slop_multiplier = 1.0 + 0.4 * x * x;

    let overall = (base_voice_distance * slop_multiplier).clamp(0.0, 1.0);

    DistanceReport {
        overall,
        base_voice_distance,
        sentence_length_dist,
        function_word_cos,
        terminal_punct_dist,
        structural_punct_dist,
        ngram_cos,
        readability_diff,
        richness_diff,
        slop_score,
        slop_multiplier,
    }
}

fn sentence_length_divergence(text: &str, fp: &StylometricFingerprint) -> f64 {
    let text_stats = lengths::sentence_lengths(text);
    if fp.sentence_length.mean == 0.0 || text_stats.mean == 0.0 {
        return 0.5;
    }

    // Use EMD when fingerprint has histogram (new fingerprints), fall back to mean/SD
    if !fp.sentence_length.histogram.is_empty() {
        let text_hist = lengths::sentence_length_histogram(text);
        return lengths::histogram_emd(&text_hist, &fp.sentence_length.histogram);
    }

    // Legacy fallback: simplified divergence using mean and SD
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

/// Split punctuation into terminal (questions, exclamations) and structural
/// (em-dashes, semicolons, colons, parentheses). en_dashes excluded as noise-prone.
///
/// Returns (terminal_distance, structural_distance), each in [0.0, 1.0].
fn punctuation_distance_split(text: &str, fp: &StylometricFingerprint) -> (f64, f64) {
    let text_punct = PunctuationStats::compute(text);
    let fp_p = &fp.punctuation;

    // Terminal: questions + exclamations — the biggest Adams gap
    let terminal_diffs = [
        (text_punct.questions_per_1k - fp_p.questions_per_1k).abs() / 10.0,
        (text_punct.exclamations_per_1k - fp_p.exclamations_per_1k).abs() / 10.0,
    ];
    let terminal = (terminal_diffs.iter().sum::<f64>() / terminal_diffs.len() as f64).clamp(0.0, 1.0);

    // Structural: em-dashes, semicolons, colons, parens (en_dashes excluded)
    let structural_diffs = [
        (text_punct.em_dashes_per_1k - fp_p.em_dashes_per_1k).abs() / 20.0,
        (text_punct.semicolons_per_1k - fp_p.semicolons_per_1k).abs() / 10.0,
        (text_punct.colons_per_1k - fp_p.colons_per_1k).abs() / 10.0,
        (text_punct.parentheses_per_1k - fp_p.parentheses_per_1k).abs() / 20.0,
    ];
    let structural = (structural_diffs.iter().sum::<f64>() / structural_diffs.len() as f64).clamp(0.0, 1.0);

    (terminal, structural)
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

/// Canonical slop scorer — single implementation used by scoring, filter, eval.
///
/// Returns a normalized 0.0-1.0 score based on density of AI-tell words/phrases.
/// Author-valid discourse markers (furthermore, moreover, nevertheless) are
/// excluded because they appear in the Adams fingerprint.
pub fn compute_slop_score(text: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let word_count = text.unicode_words().count() as f64;
    if word_count == 0.0 {
        return 0.0;
    }

    // Words that are banned as AI slop BUT present in the author corpus.
    // These must not be penalized. GPT Pro round 3 flagged this conflict.
    const AUTHOR_VALID: &[&str] = &["furthermore", "moreover", "nevertheless"];

    let mut word_occurrences: usize = 0;
    for word in ai_slop::BANNED_WORDS {
        if AUTHOR_VALID.contains(word) {
            continue;
        }
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
