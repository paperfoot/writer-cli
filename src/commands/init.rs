use serde::Serialize;

use crate::config::{self, AppConfig};
use crate::error::AppError;
use crate::output::{self, Ctx};

#[derive(Serialize)]
struct InitResult {
    config_path: String,
    profiles_dir: String,
    models_dir: String,
    active_profile: String,
    status: String,
}

pub fn run(ctx: Ctx) -> Result<(), AppError> {
    let config_dir = config::config_dir();
    let profiles_dir = config::profiles_dir();
    let models_dir = config::models_dir();
    let config_path = config::config_path();

    std::fs::create_dir_all(&config_dir)?;
    std::fs::create_dir_all(&profiles_dir)?;
    std::fs::create_dir_all(&models_dir)?;

    let default_profile_dir = profiles_dir.join("default");
    let default_samples_dir = default_profile_dir.join("samples");
    std::fs::create_dir_all(&default_samples_dir)?;

    let status = if config_path.exists() {
        "already_initialized".to_string()
    } else {
        let default_config = AppConfig::default();
        let toml = toml_serialize(&default_config)?;
        std::fs::write(&config_path, toml)?;
        "initialized".to_string()
    };

    let result = InitResult {
        config_path: config_path.display().to_string(),
        profiles_dir: profiles_dir.display().to_string(),
        models_dir: models_dir.display().to_string(),
        active_profile: "default".into(),
        status,
    };

    output::print_success_or(ctx, &result, |r| {
        use owo_colors::OwoColorize;
        println!("{} writer initialized", "+".green());
        println!("  config   {}", r.config_path.dimmed());
        println!("  profiles {}", r.profiles_dir.dimmed());
        println!("  models   {}", r.models_dir.dimmed());
        println!();
        println!("Next: {}", "writer learn <files>".bold());
    });

    Ok(())
}

fn toml_serialize(config: &AppConfig) -> Result<String, AppError> {
    Ok(format!(
        "active_profile = \"{}\"\nbase_model = \"{}\"\n\n\
         [update]\nenabled = {}\nowner = \"{}\"\nrepo = \"{}\"\n\n\
         [inference]\nbackend = \"{}\"\ntemperature = {}\nmax_tokens = {}\nollama_url = \"{}\"\n\n\
         [decoding]\nn_candidates = {}\ncontrastive_enabled = {}\ncontrastive_alpha = {}\nbanned_word_bias = {}\npreferred_word_bias = {}\nkv_quant = \"{}\"\n\n\
         [training]\nbackend = \"{}\"\nrank = {}\nalpha = {}\nlearning_rate = {}\nbatch_size = {}\nmax_steps = {}\nmax_seq_len = {}\n",
        config.active_profile,
        config.base_model,
        config.update.enabled, config.update.owner, config.update.repo,
        config.inference.backend, config.inference.temperature, config.inference.max_tokens, config.inference.ollama_url,
        config.decoding.n_candidates, config.decoding.contrastive_enabled, config.decoding.contrastive_alpha,
        config.decoding.banned_word_bias, config.decoding.preferred_word_bias, config.decoding.kv_quant,
        config.training.backend, config.training.rank, config.training.alpha,
        config.training.learning_rate, config.training.batch_size, config.training.max_steps, config.training.max_seq_len,
    ))
}
