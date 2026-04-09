<div align="center">

# writer -- Local AI that writes in your voice

**A CLI that fine-tunes a small local model on your own writing, so the text it generates sounds like you wrote it. Not like ChatGPT. Not like Claude. You.**

<br />

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/writer?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/writer/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

<br />

[![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![MSRV 1.85+](https://img.shields.io/badge/MSRV-1.85%2B-orange?style=for-the-badge)](https://www.rust-lang.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Local First](https://img.shields.io/badge/Local-First-brightgreen?style=for-the-badge)](#privacy)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=for-the-badge)](CONTRIBUTING.md)

---

Every AI writing tool sounds like every other AI writing tool. Same em-dashes. Same "Moreover, it is important to note that." Same rule-of-three bullet points. Same sycophantic open, same tidy bow at the end. The reason is simple: they all trained on the same pile of web text, so they all fall into the same attractor. `writer` goes the other way. You hand it a folder of your own writing and it fine-tunes a small model on your voice -- your vocabulary, your rhythm, your typos, your habits of punctuation. The output sounds like you because the model learned from you.

The model runs locally via [Ollama](https://ollama.com/) on Apple Silicon. Default base model is **Gemma 4 26B-A4B** (3.8B active MoE, 65-75 tok/s on M4 Max). Your writing samples never leave your laptop. No OpenAI. No Anthropic. No "we use your data to improve our services."

[Why this exists](#why-this-exists) | [Install](#install) | [Quick start](#quick-start) | [How it works](#how-it-works) | [Quality stack](#quality-stack) | [Commands](#commands) | [Privacy](#privacy)

</div>

---

## Why this exists

LLMs made text generation better and language worse. The logic improved. The prose collapsed. Every model trained on the same web-scale corpus converges on the same voice -- the same em-dashes, the same "Moreover, it is important to note that," the same rule-of-three bullet points, the same careful neutrality that offends no one and moves no one. A billion people now write with the same vocabulary, the same rhythm, the same structural habits. Language used to be rich. Now it is uniform.

The humanisation industry exists because the problem is that obvious. But humanising after the fact is a patch. You fight the model's entire training distribution on every call. And if everyone humanises the same way, the humanised output becomes the new slop -- a second layer of sameness on top of the first.

The real fix is not to disguise AI text as human. It is to bring back diversity. To capture the fingerprint of a specific voice -- yours, or a writer you admire -- and generate from *that* distribution instead of the generic one. Douglas Adams did not write like Hemingway. Hemingway did not write like Woolf. The richness of language was never about correctness. It was about difference.

That is what `writer` does. Feed it a corpus of text in a specific voice. It extracts a stylometric fingerprint -- sentence rhythm, vocabulary preferences, punctuation habits, the words the writer reaches for and the words they never use. It fine-tunes a small local model on that fingerprint. From then on, every `writer write "..."` call is a sample from *that* distribution. Not generic. Not humanised-generic. Distinctive.

You can have multiple profiles. One trained on your own blog posts. One trained on Adams. One on your company's house style. Switch between them with `writer profile use <name>`. The model stays small, runs locally, and your writing samples never leave your machine.

## Install

`writer` is a single static Rust binary. Pick one:

```bash
# From source (works today)
cargo install --git https://github.com/199-biotechnologies/writer

# From crates.io (package name is `writer-cli` because `writer` is squatted)
cargo install writer-cli

# Homebrew (planned)
brew install 199-biotechnologies/tap/writer
```

After install, the binary on your PATH is `writer`. You also need [Ollama](https://ollama.com/) and Python with [mlx-lm](https://github.com/ml-explore/mlx-lm) (for training and adapter inference on Apple Silicon):

```bash
brew install ollama && brew services start ollama
ollama pull gemma4:26b  # default base model (17 GB, MoE with 3.8B active params)
pip install mlx-lm     # for LoRA training and adapter-based generation
```

## Quick start

```bash
# 1. Create config + data directories and the default voice profile
writer init

# 2. Feed it your writing — Obsidian vaults, markdown, plain text
writer learn ~/Library/Mobile\ Documents/iCloud~md~obsidian/Documents/MyVault/
writer learn ~/Documents/drafts/*.md

# 3. Inspect the stylometric fingerprint
writer profile show

# 4. Fine-tune a LoRA adapter on your samples
writer train

# 5. Generate text in your voice
writer write "an essay about why I stopped using Twitter"
writer rewrite draft.md > revised.md
```

## How it works

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   your writing    stylometric    LoRA       quality        text that   │
│   samples    -->  fingerprint  + fine   --> decoding   --> sounds      │
│   (md, txt,       (9 feature    tune       pipeline       like you    │
│    obsidian)      categories)   (SFT       (logit bias,               │
│                                 + DPO)     rank-N,                    │
│                                            contrastive)               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Ingest** -- `writer learn` walks your directories, detects format (Markdown, Obsidian vault, plain text), strips YAML front matter, wikilinks, signatures, tracking params, and normalises the content. Deduplicates by content hash. Chunks long samples respecting paragraph boundaries.

2. **Fingerprint** -- Computes a 9-category stylometric profile grounded in authorship attribution research:
   - Word/sentence/paragraph length distributions
   - Function word frequency (200-word list, top signal per PAN shared tasks 2011-2025)
   - Character n-gram profiles (3-grams, per PAN evaluation data)
   - Punctuation patterns (em-dashes are the strongest AI tell)
   - Readability metrics (Flesch-Kincaid, Coleman-Liau, ARI)
   - Vocabulary richness (Yule's K, hapax legomena, Simpson's D)
   - AI-slop detection (70 banned words + 30 banned phrases)

3. **Train** -- LoRA fine-tuning via [mlx-tune](https://github.com/ARahim3/mlx-tune) on Apple Silicon. Optional DPO against AI-rewrites of your own samples (teaches the model what NOT to do).

4. **Generate** -- Multi-candidate generation via Ollama, ranked by stylometric distance to your fingerprint. Logit bias suppresses AI-tell words. Post-hoc filter rejects outputs that fail structural checks.

## Quality stack

Quality is opt-out, not opt-in. Every `writer write` runs the full stack:

| Layer | What it does | Reference |
|---|---|---|
| **Logit bias** | Suppress AI-tell words, boost user's preferred vocabulary | Writeprints (Abbasi & Chen, 2008) |
| **Generate-N-rank** | Generate 8 candidates, rank by stylometric distance | PAN authorship verification |
| **Post-hoc filter** | Hard-reject outputs with banned words or wrong rhythm | Custom |
| **Contrastive decoding** | Subtract base model logits from fine-tuned | CoPe (EMNLP 2025) |
| **DPO training** | Train against AI rewrites of user's own samples | Rafailov et al. (2023) |
| **Benchmark loop** | VoiceFidelity + SlopScore + CreativeWritingRubric | Custom autoresearch |

Stylometric features validated against:
- PAN Shared Tasks on Authorship Verification (2011-2025) -- function words + char n-grams as top discriminators
- Writeprints (Abbasi & Chen, ACM TOIS 2008) -- hapax legomena, POS patterns
- Yule's K (1944), Simpson's D (1949) -- vocabulary richness measures
- "Layered Insights" (EMNLP 2025) -- hybrid interpretable + neural outperforms either alone
- LUAR (Rivera-Soto et al., EMNLP 2021) -- neural authorship embeddings (planned)

## Commands

```
writer init                          First-time setup
writer learn <files>                 Ingest writing samples (md, txt, Obsidian vaults)
writer profile show                  Show the active profile's fingerprint
writer profile list                  List all profiles
writer profile new <name>            Create a new profile
writer profile use <name>            Switch active profile
writer train [--profile <name>]      Fine-tune a LoRA adapter
writer write "<prompt>"              Generate in the active voice
  --max-tokens 2048                    Control output length
  -n 4                                 Generate N candidates, return best
  -v                                   Include distance/model/timing in JSON
writer rewrite <file> [--in-place]   Rewrite a file in the active voice
writer model list                    List available base models
writer model pull <name>             Download a base model
writer config show                   Show effective configuration
writer config path                   Show config file path
writer agent-info                    Machine-readable capability manifest
writer skill install                 Install SKILL.md to agent platforms
writer update [--check]              Self-update from GitHub Releases
```

Every command accepts `--json` and `--quiet`. Default `--json` output for `writer write` is just `{"text": "..."}` — clean for piping and agent consumption. Use `-v` for detailed metrics.

## Results

Case study: Douglas Adams' complete works (684K words, 9 books, Gemma 4 26B + LoRA 500 steps).

| Condition | Stylometric Distance | What it means |
|-----------|---------------------|---------------|
| Held-out Adams text | 0.121 | Ceiling — real Adams scored against Adams fingerprint |
| **writer** (LoRA + ranked) | 0.241 | Best generated output |
| Base model (no adapter) | 0.306 | Generic Gemma 4 output |

The system closes 35% of the gap between generic LLM output and the target voice. See the [research paper](docs/paper/draft-v1.md) for full methodology and analysis.

## Agent-friendly by default

`writer` is built on the [agent-cli-framework](https://github.com/199-biotechnologies/agent-cli-framework). AI agents discover it with `writer agent-info`, learn your voice with `writer learn`, and draft in your voice from then on.

## Privacy

Everything stays local. That is the whole point.

| What | Where | Leaves your machine? |
|---|---|---|
| Writing samples | `~/Library/Application Support/writer/profiles/<name>/samples/` | No |
| Stylometric fingerprint | Computed in-process | No |
| Base model weights | Managed by Ollama | Only during initial download |
| Fine-tuned adapter | `profiles/<name>/adapters/adapters.safetensors` | No |
| Generated text | stdout | No |
| Telemetry | None | -- |

No API keys. No accounts. No upload step. You can run `writer` on an airplane.

## Configuration

`~/Library/Application Support/writer/config.toml`:

```toml
active_profile = "default"
base_model = "google/gemma-4-26b"

[inference]
backend = "ollama"             # ollama for base, mlx auto-selected when adapter present
temperature = 0.7
max_tokens = 4096
ollama_url = "http://localhost:11434"

[decoding]
n_candidates = 2               # generate N, return stylometrically closest
max_tokens = 4096              # Gemma 4 needs >=4096 (thinking model)
contrastive_enabled = true
contrastive_alpha = 0.3
banned_word_bias = -4.0

[training]
backend = "mlx-tune"
rank = 16
alpha = 32.0
learning_rate = 0.00002        # conservative for 26B models
batch_size = 1
max_steps = 500
max_seq_len = 2048
```

Precedence: compiled defaults < TOML file < env vars (`WRITER_*`).

## Contributing

Pull requests welcome. Read `AGENTS.md` for the rules (it is short). Keep the framework contract, never add interactive prompts, every error needs a `suggestion`, and `cargo test` must pass.

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [199 Biotechnologies](https://github.com/199-biotechnologies)

<br />

**If this is useful to you:**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/writer?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/writer/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
