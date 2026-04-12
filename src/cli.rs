use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "writer",
    version,
    about = "Local AI that writes in your voice",
    long_about = "Fine-tune a small local model on your own writing samples and generate text that sounds like you. No cloud, no API keys, no training data leaks."
)]
pub struct Cli {
    /// Force JSON output even in a terminal
    #[arg(long, global = true)]
    pub json: bool,

    /// Suppress informational output
    #[arg(long, global = true)]
    pub quiet: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// First-time setup: create directories and register the default profile
    Init,

    /// Feed writing samples into the active voice profile
    Learn {
        /// Files to learn from (txt, md, or any UTF-8 text)
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },

    /// Manage voice profiles
    Profile {
        #[command(subcommand)]
        action: ProfileAction,
    },

    /// Fine-tune a LoRA adapter on the active profile's samples
    Train {
        /// Profile to train (defaults to active profile)
        #[arg(long)]
        profile: Option<String>,
    },

    /// Generate text in the active voice
    Write {
        /// Prompt describing what to write
        prompt: String,

        /// Maximum tokens to generate (default: from config, typically 4096)
        #[arg(long)]
        max_tokens: Option<u32>,

        /// Number of candidates to generate and rank (default: from config)
        #[arg(long, short = 'n')]
        candidates: Option<u16>,

        /// Include model, timing, and quality details in JSON output
        #[arg(long, short = 'v')]
        verbose: bool,

        /// Raw mode: disable system_prompt and write_prompt wrapping.
        /// Sends the prompt verbatim and skips chat template in MLX.
        /// Used for evaluation and ablation experiments.
        #[arg(long)]
        raw: bool,
    },

    /// Rewrite a file in the active voice
    Rewrite {
        /// File to rewrite
        file: PathBuf,
        /// Write changes in place instead of printing to stdout
        #[arg(long)]
        in_place: bool,
    },

    /// Evaluate style fidelity across prompts, seeds, and configurations
    EvalStyle {
        /// Path to YAML prompt suite file
        #[arg(long, default_value = "prompts.yaml")]
        suite: std::path::PathBuf,

        /// Number of seeds (generations) per prompt
        #[arg(long, default_value = "4")]
        seeds: u16,

        /// Path to adapter directory (overrides auto-detection)
        #[arg(long)]
        adapter: Option<std::path::PathBuf>,

        /// Run in raw mode (no system prompt, no prompt wrapping, no chat template)
        #[arg(long)]
        raw: bool,

        /// Output directory for JSON/CSV results
        #[arg(long, short = 'o', default_value = "eval_results")]
        output: std::path::PathBuf,
    },

    /// Run format ablation: train 3 adapters, evaluate under multiple modes
    Ablation {
        /// Training steps per adapter (default: 100 for quick ablation)
        #[arg(long, default_value = "100")]
        steps: u32,

        /// Seeds per prompt during evaluation
        #[arg(long, default_value = "4")]
        seeds: u16,

        /// Path to prompt suite (default: prompts.yaml)
        #[arg(long, default_value = "prompts.yaml")]
        suite: std::path::PathBuf,

        /// Output directory for all results
        #[arg(long, short = 'o', default_value = "ablation_results")]
        output: std::path::PathBuf,

        /// Skip training, only run evaluation (assumes adapters exist)
        #[arg(long)]
        eval_only: bool,
    },

    /// Generate preference pairs for SimPO/DPO training
    GeneratePairs {
        /// Path to prompt suite YAML
        #[arg(long, default_value = "prompts.yaml")]
        suite: std::path::PathBuf,

        /// Candidates per prompt (scored and paired)
        #[arg(long, default_value = "4")]
        candidates: u16,

        /// Seeds per prompt
        #[arg(long, default_value = "3")]
        seeds: u16,

        /// Minimum margin between chosen and rejected (stylometric distance)
        #[arg(long, default_value = "0.08")]
        min_margin: f64,

        /// Output JSONL file for preference pairs
        #[arg(long, short = 'o', default_value = "preference_pairs.jsonl")]
        output: std::path::PathBuf,
    },

    /// Train SimPO/DPO on preference pairs
    TrainDpo {
        /// Path to preference pairs JSONL
        #[arg(long, default_value = "preference_pairs.jsonl")]
        pairs: std::path::PathBuf,

        /// Preference method: simpo or dpo
        #[arg(long, default_value = "simpo")]
        method: String,

        /// Learning rate (default: 1e-6)
        #[arg(long, default_value = "1e-6")]
        lr: f32,

        /// Training steps
        #[arg(long, default_value = "500")]
        steps: u32,

        /// SimPO gamma (reward margin)
        #[arg(long, default_value = "1.0")]
        gamma: f32,

        /// DPO beta (KL penalty)
        #[arg(long, default_value = "0.1")]
        beta: f32,
    },

    /// Extract canon lexicon (named entities + high-PMI terms) from corpus
    BuildLexicon {
        /// Profile to extract from (defaults to active)
        #[arg(long)]
        profile: Option<String>,

        /// Minimum word count for a term to be included
        #[arg(long, default_value = "3")]
        min_count: usize,

        /// Maximum terms to include
        #[arg(long, default_value = "200")]
        max_terms: usize,
    },

    /// Manage base models
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Machine-readable capability manifest
    #[command(visible_alias = "info")]
    AgentInfo,

    /// Manage skill file installation for AI agent platforms
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Self-update from GitHub Releases
    Update {
        /// Check only, do not install
        #[arg(long)]
        check: bool,
    },
}

#[derive(Subcommand)]
pub enum ProfileAction {
    /// Show the active profile's stylometric fingerprint
    Show,
    /// List all saved profiles
    #[command(visible_alias = "ls")]
    List,
    /// Switch the active profile
    Use {
        /// Profile name
        name: String,
    },
    /// Create a new empty profile
    New {
        /// Profile name
        name: String,
    },
}

#[derive(Subcommand)]
pub enum ModelAction {
    /// List available base models
    #[command(visible_alias = "ls")]
    List,
    /// Download a base model
    Pull {
        /// Model name, e.g. "llama-3.2-3b-instruct"
        name: String,
    },
}

#[derive(Subcommand)]
pub enum SkillAction {
    /// Write SKILL.md to all detected agent platforms
    Install,
    /// Check which platforms have the skill installed
    Status,
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Display effective merged configuration
    Show,
    /// Print the configuration file path
    Path,
}
