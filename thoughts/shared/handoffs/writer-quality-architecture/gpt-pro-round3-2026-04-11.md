# GPT Pro Round 3 — Scoring, Eval, and Ablation Review

**Date:** 2026-04-11
**Input:** 21 source files, fingerprint, lexicon, prompt suite, session 4 handoff
**Questions:** 5 (corrections review, pre-ablation work, ablation efficiency, eval harness gaps, contamination risks)

## Critical Findings

### Scoring contamination (must fix before ablation)
1. **Slop penalizes Adams vocab:** `ai_slop.rs` bans `furthermore`, `moreover`, `nevertheless` — but Adams uses them. Fingerprint confirms `nevertheless` and `furthermore` present.
2. **Punctuation buried:** Missing all Adams questions+exclamations (8.95+5.24 per 1k → 0.0 actual) contributes only ~0.020 to distance. One slop hit contributes 0.030. The ranker over-rewards "clean but flat."
3. **Three inconsistent slop implementations:** scoring.rs (density), filter.rs (contains+count), slop_score.rs (unique). Must unify.

### Eval harness cannot support decision protocol
4. **No prompt relevance metric** — gate 2 is unmeasurable
5. **Failures hidden** — dropped from results, filter-failed returned as successes
6. **No per-bucket stats** — ablation picks winners from flat global means
7. **Seed is fake** — loop variable only, not threaded to MLX. No reproducibility.
8. **Raw confounds two factors** — raw = no system prompt AND no wrapping. Must separate.

### Ablation efficiency
9. **One-shot Python bridge is #1 waste** — loads 15GB model per generation
10. **n_candidates multiplier** — default 8 means 8x more MLX calls than needed for format comparison
11. **Minimum valid experiment:** 3×2×6×2×1×1 = 72 single-pass requests

### Fingerprint contamination
12. **Sentence segmentation suspect** — abbreviation splitting makes mean 6.8 unreliable
13. **Don't implement EMD against broken baseline** — fix splitter first, then regenerate, then EMD
14. **Holdout not randomized** — `voice_fidelity.rs` takes first N samples, not shuffled

### Codex/Gemini correction arbitration
15. **Codex dead-zone quadratic > Gemini exponential** — exponential has wrong sign for distance metric
16. **Finer bins correct** — but only after sentence splitter is fixed
17. **Split terminal/structural punctuation** — correct and necessary

## Implementation Order (GPT Pro recommended)

1. Thread true seeds through eval → decoding → MLX
2. Persistent MLX worker (one per adapter)
3. Ablation-specific overrides: n_candidates=1, max_attempts=1, separate raw/system factors
4. Per-generation JSON output from ablation (not just aggregates)
5. Bucketed reporting + hard-gate winner selection
6. Clean canon leakage metric (token boundaries, term logging, entity/generic split)
7. Audit sentence segmentation, fix, regenerate fingerprint
8. Rework scoring (split punct, unify slop, remove author-valid items, post-score penalty)
9. Sentence-length histogram + EMD with legacy fallback
10. Run ablation: 3×2×6×2 = 72 gens on 26B
11. Post-ablation: tune weights, add prompts, test system-prompt factor
