//! Rank generated candidates by multi-objective scoring.
//!
//! Per GPT Pro review (Prompt-and-Rerank, arXiv:2205.11503, PAN systems):
//! decompose into base voice distance, relevance, slop, and repetition
//! rather than a single combined score.
use crate::stylometry::fingerprint::StylometricFingerprint;
use crate::stylometry::{relevance, scoring};

use super::filter;

/// Rank candidates by multi-objective score (lower = better).
///
/// Score = base_voice_distance
///       + 0.25 * (1 - relevance)^2    — squared penalty for off-topic
///       + 0.10 * slop_score            — AI-tell penalty
///       + 0.10 * repetition_ratio      — soft repetition penalty
///
/// Returns vec of (candidate_index, combined_score), sorted lowest first.
pub fn rank(
    candidates: &[(String, u32, u64)],
    fingerprint: &StylometricFingerprint,
    prompt: &str,
) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, (text, _, _))| {
            let report = scoring::distance(text, fingerprint);
            let rel = relevance::score(prompt, text);
            let rep = filter::repetition_ratio(text);

            // Multi-objective combination
            let combined = report.base_voice_distance
                + 0.25 * (1.0 - rel).powi(2)  // squared: mild penalty for partial relevance
                + 0.10 * report.slop_score     // direct slop penalty
                + 0.10 * rep;                  // soft repetition penalty

            (i, combined.clamp(0.0, 1.5))
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
