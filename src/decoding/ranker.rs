//! Rank generated candidates by stylometric distance to the user's fingerprint.
//!
//! Reference: PAN authorship verification — cosine/distance-based ranking.
use crate::stylometry::fingerprint::StylometricFingerprint;
use crate::stylometry::scoring;

/// Rank candidates by stylometric distance to the fingerprint.
/// Returns vec of (candidate_index, distance), sorted lowest distance first.
pub fn rank(
    candidates: &[(String, u32, u64)],
    fingerprint: &StylometricFingerprint,
) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, (text, _, _))| {
            let report = scoring::distance(text, fingerprint);
            (i, report.overall)
        })
        .collect();

    // Sort by distance ascending (closest to user's voice first)
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}
