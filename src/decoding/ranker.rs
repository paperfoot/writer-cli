//! Rank generated candidates by combined style fidelity and prompt relevance.
//!
//! Reference: PAN authorship verification — cosine/distance-based ranking.
use crate::stylometry::fingerprint::StylometricFingerprint;
use crate::stylometry::{relevance, scoring};

/// Rank candidates by combined score: style distance penalized by low relevance.
///
/// Scoring: `combined = style_distance + relevance_penalty`
/// where `relevance_penalty = (1.0 - relevance) * 0.3`
///
/// This means: a perfectly relevant but stylistically distant candidate (0.6 + 0.0)
/// beats an off-topic but well-styled candidate (0.3 + 0.3).
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
            // Penalty: up to 0.3 for completely irrelevant output
            let relevance_penalty = (1.0 - rel) * 0.3;
            let combined = (report.overall + relevance_penalty).clamp(0.0, 1.0);
            (i, combined)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
