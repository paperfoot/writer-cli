//! Combined benchmark score for the autoresearch loop.
//!
//! Formula from handoff:
//! combined = (1.0 - voice_fidelity_distance) * 0.50
//!          + (1.0 - slop_score / 100.0) * 0.30
//!          + (creative_rubric_score / 100.0) * 0.20
use serde::Serialize;

use super::slop_score::SlopScoreResult;
use super::voice_fidelity::VoiceFidelityResult;

#[derive(Debug, Clone, Serialize)]
pub struct CombinedScore {
    pub combined_score: f64,
    pub voice_fidelity_distance: f64,
    pub voice_fidelity_component: f64,
    pub slop_density: f64,
    pub slop_component: f64,
    pub creative_rubric_score: f64,
    pub creative_rubric_component: f64,
}

pub fn compute(
    voice: &VoiceFidelityResult,
    slop: &SlopScoreResult,
    creative_score: f64, // 0-100, from EQ-Bench or placeholder
) -> CombinedScore {
    let voice_component = (1.0 - voice.mean_distance) * 0.50;
    let slop_component = (1.0 - slop.ai_density_score / 100.0) * 0.30;
    let creative_component = (creative_score / 100.0) * 0.20;

    let combined = (voice_component + slop_component + creative_component).clamp(0.0, 1.0);

    CombinedScore {
        combined_score: combined,
        voice_fidelity_distance: voice.mean_distance,
        voice_fidelity_component: voice_component,
        slop_density: slop.ai_density_score,
        slop_component,
        creative_rubric_score: creative_score,
        creative_rubric_component: creative_component,
    }
}
