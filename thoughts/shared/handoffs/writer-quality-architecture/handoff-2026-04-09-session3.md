# Session Handoff — writer CLI: Phases 0-5, Gemma 4 26B, LoRA verified

**Date:** 2026-04-09 23:30
**Session:** Full implementation sprint — Phases 0 through 5, end-to-end generation on Gemma 4 26B
**Context usage at handoff:** Very high — full plan + implementation + benchmarking + research

---

## Mission

Build `writer` into a local-first CLI that generates text **indistinguishable from a specific author's voice**. The product thesis: LLMs improved logic but collapsed linguistic diversity. Everyone generates the same em-dashes, the same transitions, the same structural patterns. Humanising after the fact just adds a second layer of sameness. The fix is to capture the stylometric fingerprint of a specific voice and generate from *that* distribution. Douglas Adams doesn't write like Hemingway. The richness of language was about difference, not correctness.

**Current proof-of-concept:** Douglas Adams' complete works (9 books, 684K words). The user will manually read generated output to confirm it captures Adams' voice — the metrics guide optimisation but human judgement is the final arbiter.

**Multi-voice by design:** `writer profile new boris && writer learn ~/obsidian/ && writer train`. Switch voices with `writer profile use <name>`.

---

## Quality Standards (NON-NEGOTIABLE for the next session)

1. **Every technical decision must be grounded in research.** Cite papers, repos, or benchmarks. No guessing from memory. If you're unsure about a method, search for it. Key references already found:
   - PAN Shared Tasks (2011-2025) — function words + char n-grams as top authorship signals
   - Writeprints (Abbasi & Chen, ACM TOIS 2008) — hapax legomena, vocabulary features
   - Yule (1944), Simpson (1949) — vocabulary richness measures
   - "Layered Insights" (EMNLP 2025) — hybrid interpretable + neural outperforms either alone
   - LUAR (Rivera-Soto et al., EMNLP 2021) — neural authorship embeddings
   - Binoculars (ICML 2024) — zero-shot AI text detection
   - CoPe (EMNLP 2025) — contrastive decoding for personalised LLMs

2. **Validate assumptions.** LLM APIs, model tags, library interfaces change weekly. Before calling any external tool, verify the interface. Before assuming a model supports a feature, probe it. The Gemma 4 `/api/chat` vs `/api/generate` issue cost an hour.

3. **Use Codex (GPT-5.4 with xhigh reasoning) for adversarial review** of any significant code. It found 2 HIGH bugs this session that would have corrupted benchmarks. Run it after each phase completion.

4. **Tests must prove something.** Codex found that `scoring_human_text_closer_to_human_fingerprint` fingerprints the SAME text it scores — it proves self-distance is low, nothing about generalisation. Use held-out samples. Test edge cases. If a test can't fail in a meaningful way, it's not a test.

5. **The user will manually evaluate generated text.** Metrics are a guide, not the goal. Don't benchmaxx. If the combined score is 0.95 but the text reads like generic LLM output, the tool is broken.

---

## Active Plan

**Plan file:** `docs/superpowers/plans/2026-04-09-writer-quality-architecture.md` (2,405 lines)

| Phase | Status |
|---|---|
| 0: Foundation | **Done** — tagged `phase0-complete`, 14 tests |
| 1: Data pipeline | **Done** — sources, normalize, dedupe, chunk, ingest, 8 tests |
| 2: Stylometric fingerprint v2 | **Done** — 9 evidence-based feature categories, 10 tests |
| 3: Ollama inference backend | **Done** — /api/chat, MLX 0.20.4, multi-candidate, 4 tests |
| 3.5: Benchmark harness | **Done** — `writer-bench` binary |
| 4: Quality decoding layer | **Done** — logit bias, rank-N, filter, system prompt priming |
| 5: mlx-tune LoRA backend | **Wired** — training verified (200 steps, loss 3.34→2.33), HF model path needs fix for Gemma 4 |
| 6: DPO against AI-rewrites | Not started |
| 7: Contrastive decoding (CoPe) | Not started |
| 8: `writer score` CLI | Not started |
| 9: Extensibility docs | Not started |

---

## What Was Accomplished This Session

- **Phases 0-4 fully implemented** with 47 tests across 7 test files, all passing
- **Phase 5 wired** — MlxTuneBackend spawns `mlx_lm.lora`, training data prepared (441 train + 49 valid samples in mlx-lm chat format), LoRA training verified on Gemma 3 4B (200 steps, loss 3.34→2.33, adapter generates Adams-like dialogue structure)
- **Ollama upgraded** 0.17.6 → 0.20.4 (MLX backend, flash attention, KV cache quant)
- **Gemma 4 26B** (17GB MoE) pulled and generating on M4 Max (51.8 GiB VRAM)
- **Douglas Adams corpus ingested:** 492 samples, 684,089 words, 22,653 unique vocabulary, fingerprint saved with 9 feature categories
- **Epub cleanup pipeline:** strips HTML tags, SVG blocks, pandoc divs, link anchors, image refs, escaped chars, signatures, zero-width chars, tracking params — verified clean on random sample inspection
- **README rewritten** with the real product thesis: bring back linguistic diversity, not just humanise generic output
- **Codex adversarial review** found 8 issues — 2 HIGH and 1 MEDIUM fixed this session
- **Research agent** identified key repos (LUAR, Binoculars) and missing features (POS n-grams, discourse markers, sentence-initial patterns)

---

## Benchmark Baselines

| Metric | Held-out Adams (human) | Generated (Gemma 3 4B) | Generated (Gemma 4 26B) | Target (Phase 7) |
|---|---|---|---|---|
| Voice fidelity distance | **0.121** | 0.282 | 0.297 | 0.15 |
| Slop density | 0.021% | not scored | not scored | < 0.5% |
| Combined score | **0.883** | not computed | not computed | 0.85 |

**Note:** The held-out baseline is high because it's scoring real Adams text against the Adams fingerprint. Generation baselines are higher (worse) because the model hasn't been fine-tuned yet. The gap (0.121 → 0.282) is what LoRA + DPO + contrastive decoding will close.

---

## Key Decisions Made

1. **Gemma 4 26B requires `/api/chat`, not `/api/generate`.** It's a thinking model — uses `thinking` field for chain-of-thought, `content` for final answer. Needs `num_predict >= 4096` to finish thinking + response. Empty content = token budget exhausted during thinking.

2. **System prompt priming is double-edged.** Prescriptive priming ("avg sentence length: 6.8") makes the model produce choppy conformist text. Soft guidance ("short punchy sentences mixed with longer ones") works better. Still needs tuning with the autoresearch loop.

3. **LoRA training on 4-bit quantised MLX models works.** `mlx-community/gemma-3-4b-it-4bit` trained successfully. The Gemma 4 26B version (`mlx-community/gemma-4-26b-a4b-it-4bit`) stalled at 328MB download — likely needs `huggingface-cli login` for gated model access.

4. **Normalization removes stylometric signal** (Codex finding, FIXED). `normalize_whitespace()` and `normalize_quotes()` were flattening spacing and quote style that are part of the author's fingerprint. SVG stripping also had a bug with inline blocks. Both fixed.

5. **AI-slop penalty was counting presence, not occurrences** (Codex finding, FIXED). Twenty uses of "delve" scored the same as one. Now counts actual occurrences.

6. **Multi-candidate generation on 26B is slow.** 8 candidates = ~4 minutes. Reduced to 2 for practical use. Future: use draft model for fast candidates + one 26B verification pass.

---

## Current State

- **Branch:** `main`
- **Last commit:** `ae0350d docs: rewrite 'Why this exists' — diversity over uniformity, voice preservation`
- **Uncommitted changes:** None
- **Tests:** 47 passing across 7 test files
- **Build:** Clean (warnings only)
- **Remote:** https://github.com/199-biotechnologies/writer — 23 commits pushed
- **Ollama:** 0.20.4, models: gemma4:26b (17GB), gemma3:4b (3.3GB), qwen3.5:9b (6.6GB)
- **Config:** `base_model = "google/gemma-4-26b"`, `n_candidates = 2`, `max_tokens = 4096`
- **Corpus:** Adams 492 samples in `~/Library/Application Support/writer/profiles/default/samples/corpus.jsonl`
- **Fingerprint:** `~/Library/Application Support/writer/profiles/default/fingerprint.json`
- **Training data:** `~/Library/Application Support/writer/profiles/default/training_data/{train,valid,test}.jsonl` (441 + 49 + 49)
- **LoRA adapter (Gemma 3 4B proof-of-concept):** `~/Library/Application Support/writer/profiles/default/adapters-gemma3-200/adapters.safetensors`

---

## What to Do Next (ordered)

### Immediate: wire LoRA end-to-end

1. **`huggingface-cli login`** — needed for gated Gemma 4 model weights. Run: `! huggingface-cli login` in the prompt.

2. **Download Gemma 4 MLX weights:** `python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit')"` — verify it completes.

3. **Train LoRA on Gemma 4:** 500+ steps, batch_size=1, seq_length=2048, lr=2e-5. Monitor for loss < 2.0. The Gemma 3 4B proof showed loss 3.34→2.33 in 200 steps.

4. **Wire adapter loading in generation.** When `writer write` detects an adapter at `profiles/<name>/adapters/`, pass it to Ollama via a Modelfile or pass the adapter path to `mlx_lm.generate` directly. **Research how Ollama loads LoRA adapters** — check `ollama create` with a Modelfile containing `ADAPTER /path/to/safetensors`.

5. **Re-benchmark with adapter loaded.** Compare generated text distance with and without adapter. The gap should shrink measurably.

6. **Have the user manually read 5-10 generated passages** and judge whether they sound like Adams.

### Then: autoresearch loop

7. **Wire `writer-bench` to generate text via the model** (currently it only scores held-out human text). It should generate N prompts, score each against the fingerprint, and report the mean.

8. **Start autoresearch:**
   ```bash
   autoresearch start \
     --metric "writer-bench run --json | jq -r '.data.combined_score'" \
     --baseline-command "writer-bench baseline --save" \
     --improvement-threshold 0.01 \
     --max-iterations 50 \
     --eval-timeout 900 \
     --keep-on-regression false
   ```

### Then: Phases 6-7

9. **Phase 6: DPO** — generate AI rewrites of Adams samples (rejected), pair with originals (chosen), train DPO adapter on top of LoRA SFT. **Research:** Rafailov et al. (2023) "Direct Preference Optimization" — verify mlx-lm supports DPO training (`mlx_lm.dpo` or similar).

10. **Phase 7: Contrastive decoding** — subtract base model logits from fine-tuned. **Research:** check if Ollama 0.20.4 supports `logprobs` in the chat API response. If not, use `mlx_lm.generate` with `--verbose` to get per-token logprobs. Reference: CoPe paper (EMNLP 2025, arxiv.org/html/2506.12109v2).

---

## Codex Adversarial Review Findings

| Severity | Issue | Status |
|---|---|---|
| HIGH | SVG stripping: inline `<svg>...</svg>` on one line left `in_svg` stuck | **FIXED** |
| HIGH | `candidate_index` wrong when candidates complete out-of-order from JoinSet | **FIXED** |
| MEDIUM | `GenerationEvent::Error` was fatal — one bad candidate killed all | **FIXED** |
| MEDIUM | Normalization destroyed stylometric signal (whitespace, quotes) | Partially fixed — quote handling still asymmetric |
| MEDIUM | AI-slop penalty counted presence, not occurrences | **FIXED** |
| MEDIUM | Test `sentence_lengths_differ_between_styles` is vacuous | **NOT FIXED** — needs stronger assertion |
| MEDIUM | Test `scoring_human_text_closer_to_human_fingerprint` proves nothing about generalisation | **NOT FIXED** — needs held-out test data |
| LOW | `fingerprint_serializes_roundtrip` only checks word_count | NOT FIXED |

**Next session should fix the remaining MEDIUM test issues** before adding more features.

---

## Research Findings (for grounding future work)

### Repos to integrate
- **LLNL/LUAR** (github.com/LLNL/LUAR) — transformer authorship embeddings, Apache-2.0, HuggingFace weights. Add cosine similarity of LUAR vectors as a scoring dimension.
- **ahans30/Binoculars** (github.com/ahans30/Binoculars) — ICML 2024, zero-shot AI detection using dual-LM perplexity contrast. >90% detection at 0.01% FPR.
- **shaoormunir/writeprints** — Python extraction of Writeprints features for reference.

### Missing features (evidence-based, should add)
- **POS tag n-grams** — strongest syntactic discriminator per PAN tasks. Needs spaCy or lightweight tagger.
- **Discourse markers** — "however", "actually", "basically" frequency. Strong authorship signal.
- **Sentence-initial word patterns** — distribution of sentence starters (pronouns, articles, conjunctions).
- **LUAR embeddings** — single high-signal feature alongside hand-crafted features. "Layered Insights" (EMNLP 2025) confirms hybrid outperforms either alone.

### Key papers
- "Layered Insights: Generalizable Analysis of Human Authorial Style" (EMNLP 2025) — hybrid approach
- "Can LLMs Identify Authorship?" (EMNLP 2024 Findings) — LLM-based attribution
- "Residualized Similarity for Explainable Authorship Verification" (EMNLP 2025 Findings)
- "Open-World Authorship Attribution" (ACL 2025 Findings)
- CoPe: Personalized LLM Decoding via Contrasting Personal Preference (EMNLP 2025)

---

## Douglas Adams Fingerprint (key stats for reference)

| Feature | Value | Note |
|---|---|---|
| Sentence length mean | 6.8 words | Short, punchy — classic Adams |
| Sentence length SD | 4.3 | High variance — mixes short and long |
| Word length mean | 4.44 chars | Simple vocabulary |
| Flesch-Kincaid grade | 3.94 | Very readable |
| Questions per 1k words | 8.95 | Lots of rhetorical questions |
| Exclamations per 1k | 5.24 | Expressive |
| Em-dashes per 1k | 1.28 | Moderate use |
| Yule's K | 85.33 | Rich vocabulary |
| Hapax legomena ratio | 24.7% | Many unique words |
| Vocabulary size | 22,653 | Across 684K words |
| AI-slop words NOT in Adams | 42 | (ironic number) |

---

## Files to Review First

1. `docs/superpowers/plans/2026-04-09-writer-quality-architecture.md` — the master plan (2,405 lines)
2. `src/decoding/mod.rs` — the quality pipeline
3. `src/stylometry/scoring.rs` — distance function
4. `src/backends/inference/ollama.rs` — Ollama backend (/api/chat)
5. `src/backends/training/mlx_tune.rs` — LoRA training backend
6. `benchmarks/` — baseline numbers to beat

---

## Gotchas & Warnings

1. **Gemma 4 26B is a thinking model.** Empty content = ran out of tokens during thinking. Set `max_tokens >= 4096`.
2. **Use `/api/chat` not `/api/generate`.** The backend was switched. If reverted, all generation breaks.
3. **Ollama 0.20.4** was installed alongside the old 0.17.6. New binary at `/usr/local/bin/ollama`. If Ollama regresses, check which version is running.
4. **`mlx-community/gemma-4-26b-a4b-it-4bit`** download may need HF auth. Run `huggingface-cli login` first.
5. **First sample in corpus.jsonl may have residual artifacts** from epub conversion cover page.
6. **Single-agent sessions: do NOT use TaskCreate/TaskUpdate.** Global CLAUDE.md rule.
7. **Douglas Adams corpus** is at `/Users/biobook/Projects/douglas-adamiser/DA/markdown/` (9 books). Already ingested.
8. **User's Obsidian vaults** at `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/` — 199 vault has 968K words, not yet ingested. User wants multiple voice profiles.
9. **Sentence length stats seem low (6.8 mean)** because `unicode_segmentation::unicode_sentences()` splits on abbreviation periods. Calibrate expectations accordingly.
10. **The user will manually judge generated text quality.** Metrics are a guide. Don't optimise for numbers at the expense of actually sounding like Adams.

---

## Commands to Verify State

```bash
cd /Users/biobook/Projects/writer
git status --short && git log --oneline -5
cargo test --tests 2>&1 | grep "test result"
ollama list
./target/debug/writer write "Test." --json 2>&1 | head -3
./target/debug/writer-bench run --smoke --json 2>&1 | python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print(f'Combined: {d[\"combined_score\"]:.3f}')"
```

Expected: clean tree, 47 tests passing, Gemma 4 26B in model list, generation working, combined ~0.883.
