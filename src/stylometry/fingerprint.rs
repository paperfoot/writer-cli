//! Placeholder — real computation lands in Phase 2.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StylometricFingerprint {
    pub word_count: u64,
}
