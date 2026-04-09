//! Capability advertisement for inference backends.
//!
//! Every backend returns a `BackendCapabilities` from
//! [`InferenceBackend::capabilities`]. The decoding layer inspects this
//! struct before calling `generate` to negotiate which quality features
//! are available. Flags default to off so unknown backends behave
//! conservatively.
use serde::{Deserialize, Serialize};

/// KV cache optimisation family supported by the backend.
///
/// Multiple kinds can stack (e.g. a backend may advertise `Both` if it
/// can run TurboQuant compression _and_ TriAttention pruning together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum KvQuantKind {
    #[default]
    None,
    /// Google TurboQuant / PolarQuant / QJL KV-value quantisation.
    TurboQuant,
    /// MLX TriAttention key-pruning.
    TriAttention,
    /// Both families active simultaneously.
    Both,
}

/// Weight quantisation scheme the backend can load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantSchemeKind {
    Bf16,
    Fp16,
    Q8_0,
    Q6K,
    Q5KM,
    Q4KM,
    Q3KM,
    AWQ,
    UnslothDynamic20,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supports_lora: bool,
    pub supports_logit_bias: bool,
    pub supports_contrastive_decoding: bool,
    pub supports_speculative_decoding: bool,
    pub supports_activation_steering: bool,
    pub kv_quant: KvQuantKind,
    pub quant_schemes: Vec<QuantSchemeKind>,
    pub max_context: usize,
    pub streaming: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_lora: false,
            supports_logit_bias: false,
            supports_contrastive_decoding: false,
            supports_speculative_decoding: false,
            supports_activation_steering: false,
            kv_quant: KvQuantKind::None,
            quant_schemes: vec![],
            max_context: 2048,
            streaming: false,
        }
    }
}
