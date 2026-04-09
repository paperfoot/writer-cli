/// Machine-readable capability manifest.
///
/// agent-info is always raw JSON (not wrapped in the envelope) because it
/// IS the schema definition. An agent calling agent-info is bootstrapping.
pub fn run() {
    let config_path = crate::config::config_path();
    let profiles_dir = crate::config::profiles_dir();
    let models_dir = crate::config::models_dir();

    let info = serde_json::json!({
        "name": "writer",
        "version": env!("CARGO_PKG_VERSION"),
        "description": env!("CARGO_PKG_DESCRIPTION"),
        "commands": {
            "init": {
                "description": "First-time setup: create directories and register default profile",
                "args": [],
                "options": []
            },
            "learn": {
                "description": "Feed writing samples into the active voice profile",
                "args": [
                    {
                        "name": "files",
                        "kind": "positional",
                        "type": "path[]",
                        "required": true,
                        "description": "One or more text files to ingest"
                    }
                ],
                "options": []
            },
            "profile show": {
                "description": "Show the active profile's stylometric fingerprint",
                "args": [],
                "options": []
            },
            "profile list": {
                "description": "List all saved profiles",
                "aliases": ["profile ls"],
                "args": [],
                "options": []
            },
            "profile use": {
                "description": "Switch the active profile",
                "args": [
                    { "name": "name", "kind": "positional", "type": "string", "required": true }
                ],
                "options": []
            },
            "profile new": {
                "description": "Create a new empty profile",
                "args": [
                    { "name": "name", "kind": "positional", "type": "string", "required": true }
                ],
                "options": []
            },
            "train": {
                "description": "Fine-tune a LoRA adapter on the active profile's samples",
                "args": [],
                "options": [
                    {
                        "name": "--profile",
                        "type": "string",
                        "required": false,
                        "description": "Profile to train (defaults to active)"
                    }
                ]
            },
            "write": {
                "description": "Generate text in the active voice",
                "args": [
                    {
                        "name": "prompt",
                        "kind": "positional",
                        "type": "string",
                        "required": true,
                        "description": "Prompt describing what to write"
                    }
                ],
                "options": []
            },
            "rewrite": {
                "description": "Rewrite a file in the active voice",
                "args": [
                    {
                        "name": "file",
                        "kind": "positional",
                        "type": "path",
                        "required": true
                    }
                ],
                "options": [
                    {
                        "name": "--in-place",
                        "type": "bool",
                        "required": false,
                        "default": false,
                        "description": "Write changes in place instead of stdout"
                    }
                ]
            },
            "model list": {
                "description": "List available base models",
                "aliases": ["model ls"],
                "args": [],
                "options": []
            },
            "model pull": {
                "description": "Download a base model",
                "args": [
                    { "name": "name", "kind": "positional", "type": "string", "required": true }
                ],
                "options": []
            },
            "agent-info": {
                "description": "This manifest",
                "aliases": ["info"],
                "args": [],
                "options": []
            },
            "skill install": {
                "description": "Install skill file to agent platforms",
                "args": [],
                "options": []
            },
            "skill status": {
                "description": "Check skill installation status",
                "args": [],
                "options": []
            },
            "config show": {
                "description": "Display effective merged configuration",
                "args": [],
                "options": []
            },
            "config path": {
                "description": "Show configuration file path",
                "args": [],
                "options": []
            },
            "update": {
                "description": "Self-update from GitHub Releases",
                "args": [],
                "options": [
                    {
                        "name": "--check",
                        "type": "bool",
                        "required": false,
                        "default": false,
                        "description": "Check only, do not install"
                    }
                ]
            }
        },
        "global_flags": {
            "--json": {
                "description": "Force JSON output (auto-enabled when piped)",
                "type": "bool",
                "default": false
            },
            "--quiet": {
                "description": "Suppress informational output",
                "type": "bool",
                "default": false
            }
        },
        "exit_codes": {
            "0": "Success",
            "1": "Transient error (IO, network) -- retry",
            "2": "Config error -- fix setup",
            "3": "Bad input -- fix arguments",
            "4": "Rate limited -- wait and retry"
        },
        "envelope": {
            "version": "1",
            "success": "{ version, status, data }",
            "error": "{ version, status, error: { code, message, suggestion } }"
        },
        "config": {
            "path": config_path.display().to_string(),
            "profiles_dir": profiles_dir.display().to_string(),
            "models_dir": models_dir.display().to_string(),
            "env_prefix": "WRITER_",
            "inference_backend": "ollama",
            "training_backend": "mlx-tune",
            "decoding_stack": [
                "logit_bias_from_fingerprint",
                "contrastive_decoding_optional",
                "generate_n_rank",
                "post_hoc_filter"
            ]
        },
        "auto_json_when_piped": true
    });
    println!("{}", serde_json::to_string_pretty(&info).unwrap());
}
