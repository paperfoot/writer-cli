//! writer -- local AI that writes in your voice.
//!
//! Built on the agent-cli-framework. Every command returns a JSON envelope
//! when piped, a colored table in a terminal, and exits with a semantic
//! code in 0-4. `agent-info` describes the whole surface.

mod cli;
mod commands;
mod output;

use writer_cli::config;
use writer_cli::error;

use clap::Parser;

use cli::{Cli, Commands, ConfigAction, ModelAction, ProfileAction, SkillAction};
use output::{Ctx, Format};

/// Pre-scan argv for --json before clap parses. This ensures --json is
/// honored on help, version, and parse-error paths where clap has not
/// populated the Cli struct yet.
fn has_json_flag() -> bool {
    std::env::args_os().any(|a| a == "--json")
}

#[tokio::main]
async fn main() {
    let json_flag = has_json_flag();

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            // Help and --version are not errors. Exit 0.
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                let format = Format::detect(json_flag);
                match format {
                    Format::Json => {
                        output::print_help_json(e);
                        std::process::exit(0);
                    }
                    Format::Human => e.exit(),
                }
            }

            // Parse errors -- we own the exit code, never clap. Always 3.
            let format = Format::detect(json_flag);
            output::print_clap_error(format, &e);
            std::process::exit(3);
        }
    };

    let ctx = Ctx::new(cli.json, cli.quiet);

    let result = match cli.command {
        Commands::Init => commands::init::run(ctx),
        Commands::Learn { files } => commands::learn::run(ctx, files), // TODO: async ingest in future
        Commands::Profile { action } => match action {
            ProfileAction::Show => commands::profile::show(ctx),
            ProfileAction::List => commands::profile::list(ctx),
            ProfileAction::Use { name } => commands::profile::use_profile(ctx, name),
            ProfileAction::New { name } => commands::profile::new_profile(ctx, name),
        },
        Commands::Train { profile } => commands::train::run(ctx, profile).await,
        Commands::Write {
            prompt,
            max_tokens,
            candidates,
            verbose,
            raw,
        } => commands::write::run(ctx, prompt, max_tokens, candidates, verbose, raw).await,
        Commands::EvalStyle {
            suite,
            seeds,
            adapter,
            raw,
            output,
        } => commands::eval_style::run(ctx, suite, seeds, adapter, raw, output).await,
        Commands::GeneratePairs {
            suite,
            candidates,
            seeds,
            min_margin,
            output,
        } => commands::generate_pairs::run(ctx, suite, candidates, seeds, min_margin, output).await,
        Commands::TrainDpo {
            pairs,
            method,
            lr,
            steps,
            gamma,
            beta,
        } => commands::train_dpo::run(ctx, pairs, method, lr, steps, gamma, beta).await,
        Commands::BuildLexicon {
            profile,
            min_count,
            max_terms,
        } => commands::build_lexicon::run(ctx, profile, min_count, max_terms),
        Commands::Ablation {
            steps,
            seeds,
            suite,
            output,
            eval_only,
        } => commands::ablation::run(ctx, steps, seeds, suite, output, eval_only).await,
        Commands::Rewrite { file, in_place } => commands::rewrite::run(ctx, file, in_place).await,
        Commands::Model { action } => match action {
            ModelAction::List => commands::model::list(ctx).await,
            ModelAction::Pull { name } => commands::model::pull(ctx, name).await,
        },
        Commands::AgentInfo => {
            commands::agent_info::run();
            Ok(())
        }
        Commands::Skill { action } => match action {
            SkillAction::Install => commands::skill::install(ctx),
            SkillAction::Status => commands::skill::status(ctx),
        },
        Commands::Config { action } => match action {
            ConfigAction::Show => config::load().and_then(|cfg| commands::config::show(ctx, &cfg)),
            ConfigAction::Path => commands::config::path(ctx),
        },
        Commands::Update { check } => {
            config::load().and_then(|cfg| commands::update::run(ctx, check, &cfg))
        }
    };

    if let Err(e) = result {
        output::print_error(ctx.format, &e);
        std::process::exit(e.exit_code());
    }
}
