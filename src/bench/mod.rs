//! Benchmark harness for the autoresearch loop.
//!
//! Three benchmarks, combined into one optimisable metric:
//! 1. VoiceFidelity (0.50) — stylometric distance to held-out user samples
//! 2. SlopScore (0.30) — AI pattern density (humanise-text detector)
//! 3. CreativeWritingRubric (0.20) — prose quality (EQ-Bench subset)
pub mod combined;
pub mod slop_score;
pub mod voice_fidelity;
