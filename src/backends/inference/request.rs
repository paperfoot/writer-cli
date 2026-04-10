//! Inference request vocabulary.
//!
//! `GenerationRequest` is the single struct every `InferenceBackend`
//! receives from the decoding layer. New features — contrastive
//! decoding, speculative decoding, KV quant preferences — are added as
//! new fields on `GenerationParams`, never as parallel code paths.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::capabilities::KvQuantKind;
use crate::backends::types::{AdapterRef, ModelId};

pub type LogitBiasMap = HashMap<String, f32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub model: ModelId,
    pub prompt: String,
    pub adapter: Option<AdapterRef>,
    pub params: GenerationParams,
    pub logit_bias: LogitBiasMap,
    pub stop_sequences: Vec<String>,
    pub system_prompt: Option<String>,
    /// "chat" (default) or "raw". In raw mode the MLX bridge skips
    /// chat template formatting and sends the prompt verbatim.
    pub prompt_mode: Option<String>,
    pub draft_model: Option<ModelId>,
    pub contrastive_base: Option<ModelId>,
}

impl GenerationRequest {
    pub fn new(model: ModelId, prompt: String) -> Self {
        Self {
            model,
            prompt,
            adapter: None,
            params: GenerationParams::default(),
            logit_bias: HashMap::new(),
            stop_sequences: Vec::new(),
            system_prompt: None,
            prompt_mode: None,
            draft_model: None,
            contrastive_base: None,
        }
    }

    pub fn with_adapter(mut self, adapter: AdapterRef) -> Self {
        self.adapter = Some(adapter);
        self
    }

    pub fn with_logit_bias(mut self, bias: LogitBiasMap) -> Self {
        self.logit_bias = bias;
        self
    }

    pub fn with_n_candidates(mut self, n: u16) -> Self {
        self.params.n_candidates = n;
        self
    }

    pub fn with_contrastive_base(mut self, base: ModelId) -> Self {
        self.contrastive_base = Some(base);
        self
    }

    pub fn with_draft_model(mut self, draft: ModelId) -> Self {
        self.draft_model = Some(draft);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationParams {
    pub n_candidates: u16,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    pub repetition_penalty: f32,
    pub contrastive_alpha: f32,
    pub kv_quant_preference: Option<KvQuantKind>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        // Quality-first defaults. The decoding layer ranks N candidates
        // and returns the best. Temperature and top_p match the setting
        // that produces the widest style variance without drifting from
        // the fine-tuned distribution.
        Self {
            n_candidates: 8,
            temperature: 0.7,
            top_p: 0.92,
            top_k: 64,
            max_tokens: 2048,
            repetition_penalty: 1.05,
            contrastive_alpha: 0.0,
            kv_quant_preference: None,
        }
    }
}
