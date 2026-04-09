//! VoiceFidelity benchmark — stylometric distance to held-out user samples.
//!
//! Weight: 0.50 of combined score.
//! Lower distance = better voice fidelity.
use crate::corpus::sample::Sample;
use crate::stylometry::fingerprint::StylometricFingerprint;
use crate::stylometry::scoring;

pub struct VoiceFidelityResult {
    pub mean_distance: f64,
    pub per_sample: Vec<f64>,
    pub n_samples: usize,
}

/// Compute mean stylometric distance between generated texts and user fingerprint.
pub fn evaluate(
    generated_texts: &[String],
    fingerprint: &StylometricFingerprint,
) -> VoiceFidelityResult {
    if generated_texts.is_empty() {
        return VoiceFidelityResult {
            mean_distance: 1.0,
            per_sample: vec![],
            n_samples: 0,
        };
    }

    let distances: Vec<f64> = generated_texts
        .iter()
        .map(|text| scoring::distance(text, fingerprint).overall)
        .collect();

    let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;

    VoiceFidelityResult {
        mean_distance,
        per_sample: distances,
        n_samples: generated_texts.len(),
    }
}

/// Split samples into training (90%) and held-out (10%) sets.
pub fn holdout_split(samples: &[Sample], holdout_ratio: f64) -> (Vec<Sample>, Vec<Sample>) {
    let n_holdout = (samples.len() as f64 * holdout_ratio).ceil() as usize;
    let n_holdout = n_holdout.max(1).min(samples.len());

    let (holdout, train) = samples.split_at(n_holdout);
    (train.to_vec(), holdout.to_vec())
}
