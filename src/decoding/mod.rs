//! Quality decoding layer — wraps every InferenceBackend::generate call.
//!
//! No `writer write` ever skips this layer. Quality is opt-out, not opt-in.
//!
//! Pipeline:
//! 1. Build prompt with stylometric priming
//! 2. Construct logit bias from fingerprint
//! 3. Generate N candidates via backend
//! 4. Rank candidates by stylometric distance
//! 5. Filter top candidate for structural quality
//! 6. Return best or regenerate
//!
//! References:
//! - PAN Shared Tasks (2011-2025) — ranking by stylometric distance
//! - Writeprints (Abbasi & Chen, 2008) — vocabulary bias
//! - CoPe (EMNLP 2025) — contrastive decoding (Phase 7)
pub mod filter;
pub mod logit_bias;
pub mod prompts;
pub mod ranker;

use serde::Serialize;
use tokio_stream::StreamExt;

use crate::backends::inference::InferenceBackend;
use crate::backends::inference::request::GenerationRequest;
use crate::backends::inference::response::GenerationEvent;
use crate::backends::types::{AdapterRef, ModelHandle, ModelId};
use crate::config::DecodingConfig;
use crate::stylometry::fingerprint::StylometricFingerprint;

#[derive(Debug, Clone, Serialize)]
pub struct GenerationResult {
    pub text: String,
    pub distance: f64,
    pub candidate_index: usize,
    pub candidates_generated: usize,
    pub tokens_generated: u32,
    pub elapsed_ms: u64,
    pub regenerations: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodingError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("all candidates rejected after {0} attempts")]
    AllRejected(usize),
}

/// Run the full quality decoding pipeline.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    backend: &dyn InferenceBackend,
    handle: &ModelHandle,
    model_id: &ModelId,
    fingerprint: &StylometricFingerprint,
    config: &DecodingConfig,
    prompt: &str,
    system_prompt: Option<&str>,
    adapter: Option<&AdapterRef>,
    prompt_mode: Option<&str>,
    seed: Option<u64>,
) -> Result<GenerationResult, DecodingError> {
    let max_attempts = config.max_attempts.unwrap_or(3) as usize;

    for attempt in 0..max_attempts {
        // Build request with logit bias from fingerprint
        let bias = logit_bias::from_fingerprint(fingerprint, config);

        let mut req = GenerationRequest::new(model_id.clone(), prompt.to_string())
            .with_logit_bias(bias)
            .with_n_candidates(config.n_candidates);
        req.params.max_tokens = config.max_tokens;
        req.params.repetition_penalty = config.repetition_penalty;

        if let Some(adapter) = adapter {
            req = req.with_adapter(adapter.clone());
        }

        if let Some(sys) = system_prompt {
            req.system_prompt = Some(sys.to_string());
        }

        if let Some(mode) = prompt_mode {
            req.prompt_mode = Some(mode.to_string());
        }

        req.seed = seed;

        // Generate candidates
        let mut stream = backend
            .generate(handle, req)
            .await
            .map_err(|e| DecodingError::Backend(e.to_string()))?;

        // Collect candidates keyed by their backend candidate_index
        // so ordering is deterministic regardless of JoinSet completion order.
        let mut candidate_map: std::collections::BTreeMap<u16, (String, u32, u64)> =
            std::collections::BTreeMap::new();
        let mut current_texts: std::collections::HashMap<u16, String> =
            std::collections::HashMap::new();
        let mut errors: Vec<String> = Vec::new();

        while let Some(event) = stream.next().await {
            match event {
                GenerationEvent::Token {
                    candidate_index,
                    text,
                    ..
                } => {
                    current_texts
                        .entry(candidate_index)
                        .or_default()
                        .push_str(&text);
                }
                GenerationEvent::Done {
                    candidate_index,
                    full_text,
                    usage,
                    ..
                } => {
                    let text = if full_text.is_empty() {
                        current_texts.remove(&candidate_index).unwrap_or_default()
                    } else {
                        full_text
                    };
                    candidate_map.insert(
                        candidate_index,
                        (text, usage.generated_tokens, usage.elapsed_ms),
                    );
                }
                GenerationEvent::Error {
                    candidate_index,
                    message,
                } => {
                    // Don't fail globally — one bad candidate shouldn't
                    // discard other usable candidates.
                    errors.push(format!("candidate {candidate_index}: {message}"));
                }
            }
        }

        // Convert to ordered vec — keys are the real backend indices
        let backend_indices: Vec<u16> = candidate_map.keys().copied().collect();
        let candidates: Vec<(String, u32, u64)> = candidate_map.into_values().collect();

        if candidates.is_empty() {
            let err_detail = if errors.is_empty() {
                "No candidates generated".to_string()
            } else {
                format!("All candidates failed: {}", errors.join("; "))
            };
            return Err(DecodingError::Backend(err_detail));
        }

        // Rank candidates by style distance + prompt relevance
        let ranked = ranker::rank(&candidates, fingerprint, prompt);

        // Filter best candidate
        let (best_vec_idx, best_distance) = ranked[0];
        let (ref best_text, tokens, elapsed) = candidates[best_vec_idx];
        // Map back to the real backend candidate index
        let best_backend_idx = backend_indices[best_vec_idx] as usize;

        let filter_result = filter::check(best_text, fingerprint, config);

        if filter_result.passed || attempt == max_attempts - 1 {
            return Ok(GenerationResult {
                text: best_text.clone(),
                distance: best_distance,
                candidate_index: best_backend_idx,
                candidates_generated: candidates.len(),
                tokens_generated: tokens,
                elapsed_ms: elapsed,
                regenerations: attempt,
            });
        }
        // Otherwise, loop and regenerate
    }

    Err(DecodingError::AllRejected(max_attempts))
}
