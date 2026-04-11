//! Word, sentence, and paragraph length distributions.
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LengthStats {
    pub mean: f64,
    pub sd: f64,
    pub median: f64,
    pub p95: f64,
    /// Histogram of lengths in fixed bins, for EMD computation.
    /// Bins: [1-2, 3-4, 5-6, 7-8, 9-10, 11-13, 14-17, 18-24, 25-34, 35-49, 50+]
    /// Values are proportions (sum to 1.0). Empty = legacy fingerprint.
    #[serde(default)]
    pub histogram: Vec<f64>,
}

/// Bin edges for sentence-length histogram.
/// Each pair (lo, hi) is inclusive. Last bin is open-ended (50+).
pub const HISTOGRAM_BINS: &[(u32, u32)] = &[
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 13),
    (14, 17),
    (18, 24),
    (25, 34),
    (35, 49),
    (50, u32::MAX),
];

/// Bin center positions for EMD transport cost.
pub const HISTOGRAM_BIN_CENTERS: &[f64] =
    &[1.5, 3.5, 5.5, 7.5, 9.5, 12.0, 15.5, 21.0, 29.5, 42.0, 60.0];

/// Compute sentence-length histogram as proportions over fixed bins.
pub fn sentence_length_histogram(text: &str) -> Vec<f64> {
    use unicode_segmentation::UnicodeSegmentation;

    let lengths: Vec<u32> = super::sentences::split_sentences(text)
        .iter()
        .map(|s| s.unicode_words().count() as u32)
        .filter(|&l| l > 0)
        .collect();

    if lengths.is_empty() {
        return vec![0.0; HISTOGRAM_BINS.len()];
    }

    let mut bins = vec![0u32; HISTOGRAM_BINS.len()];
    for &len in &lengths {
        for (i, &(lo, hi)) in HISTOGRAM_BINS.iter().enumerate() {
            if len >= lo && len <= hi {
                bins[i] += 1;
                break;
            }
        }
    }

    let total = lengths.len() as f64;
    bins.iter().map(|&c| c as f64 / total).collect()
}

/// 1D Earth Mover's Distance between two histograms.
/// For 1D distributions this equals the sum of absolute CDF differences,
/// weighted by bin-center transport distances.
pub fn histogram_emd(a: &[f64], b: &[f64]) -> f64 {
    let n_bins = HISTOGRAM_BINS.len();
    if a.len() != n_bins || b.len() != n_bins {
        return 0.5; // fallback for missing/mismatched/corrupt histograms
    }

    // Cumulative distribution difference, weighted by distance between adjacent bin centers
    let mut cdf_diff = 0.0f64;
    let mut emd = 0.0f64;

    for i in 0..a.len() {
        cdf_diff += a[i] - b[i];
        if i + 1 < HISTOGRAM_BIN_CENTERS.len() {
            let transport_cost = HISTOGRAM_BIN_CENTERS[i + 1] - HISTOGRAM_BIN_CENTERS[i];
            emd += cdf_diff.abs() * transport_cost;
        }
    }

    // Normalize: max possible EMD depends on bin spread.
    // Normalize by the max transport distance for interpretability.
    let max_transport = HISTOGRAM_BIN_CENTERS.last().unwrap_or(&60.0)
        - HISTOGRAM_BIN_CENTERS.first().unwrap_or(&1.5);
    (emd / max_transport).clamp(0.0, 1.0)
}

impl LengthStats {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let sd = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
        let p95 = sorted[p95_idx.min(sorted.len() - 1)];

        Self {
            mean,
            sd,
            median,
            p95,
            histogram: Vec::new(),
        }
    }
}

pub fn word_lengths(text: &str) -> LengthStats {
    let lengths: Vec<f64> = text
        .unicode_words()
        .map(|w| w.chars().count() as f64)
        .collect();
    LengthStats::from_values(&lengths)
}

pub fn sentence_lengths(text: &str) -> LengthStats {
    let lengths: Vec<f64> = super::sentences::split_sentences(text)
        .iter()
        .map(|s| s.unicode_words().count() as f64)
        .filter(|&l| l > 0.0)
        .collect();
    LengthStats::from_values(&lengths)
}

pub fn paragraph_lengths(text: &str) -> LengthStats {
    let lengths: Vec<f64> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.unicode_words().count() as f64)
        .filter(|&l| l > 0.0)
        .collect();
    LengthStats::from_values(&lengths)
}
