# AGENTS.md -- Instructions for AI Agents

You are working inside `writer`, a Rust CLI built on the [agent-cli-framework](https://github.com/paperfoot/agent-cli-framework). Follow these rules.

## Spirit

`writer` is a local-first AI tool. Every writing sample, every model, every fine-tuned adapter stays on the user's machine. No cloud calls. No telemetry. No training data leaves the laptop. If a change would require sending user data off-box, it does not belong in this repo.

## Architecture

```
src/
  main.rs         # entry point: parse, detect format, dispatch, exit
  cli.rs          # clap derive: Cli + Commands enums
  config.rs       # AppConfig + load() via figment (3-tier)
  error.rs        # AppError enum: exit_code + error_code + suggestion
  output.rs       # Format + Ctx + print_success_or + print_error
  commands/
    mod.rs
    agent_info.rs # capability manifest
    skill.rs      # skill install + status
    config.rs     # config show + path
    update.rs     # self-update
    init.rs       # directory bootstrap
    learn.rs      # ingest writing samples
    profile.rs    # voice profile management
    train.rs      # fine-tune (stub in v0.1)
    write.rs      # generate (stub in v0.1)
    rewrite.rs    # rewrite (stub in v0.1)
    model.rs      # base model management
```

## Non-negotiable rules

1. **JSON envelope or colored terminal output -- always.** No raw text leaks to stdout.
2. **`--help` and `--version` exit 0.** They are not errors.
3. **Errors go to stderr.** Both JSON and human-readable.
4. **Exit codes are `0, 1, 2, 3, 4`.** Nothing else.
5. **Every error has a tested `suggestion()`.** Agents follow them literally.
6. **No interactive prompts.** No stdin reads. Destructive operations take `--confirm`.
7. **`agent-info` matches reality.** Add new commands to the manifest in the same PR.

## Exit codes

| Code | Meaning | Agent action |
|------|---------|-------------|
| `0` | Success | Continue |
| `1` | Transient error (IO, network) | Retry with backoff |
| `2` | Config error (missing key, bad file) | Fix setup, do not retry |
| `3` | Bad input (invalid args) | Fix arguments |
| `4` | Rate limited | Wait, then retry |

## Directory convention

| Purpose | Path | Deletable? |
|---------|------|-----------|
| Config | `~/.config/writer/` | No (user settings) |
| Profiles | `~/.local/share/writer/profiles/` | No (user data) |
| Models | `~/.local/share/writer/models/` | Yes (cache) |
| Cache | `~/.cache/writer/` | Always safe |

## Where to put code

- New user-facing command -> new file in `src/commands/`, add variant to `cli.rs`, dispatch in `main.rs`, add entry in `agent_info.rs`.
- New config field -> add to the right struct in `config.rs`, provide a sensible default, document in README.
- New error category -> new variant in `error.rs`, wire `exit_code`, `error_code`, `suggestion`.
- Nothing goes directly in `main.rs` except the dispatch table.

## Testing

```bash
cargo build
cargo test
./target/debug/writer agent-info | jq .
./target/debug/writer --version
./target/debug/writer init
```

## Reference

- [agent-cli-framework README](https://github.com/paperfoot/agent-cli-framework) -- the full pattern catalogue
- [agent-cli-framework AGENTS.md](https://github.com/paperfoot/agent-cli-framework/blob/main/AGENTS.md) -- the framework rules this repo inherits
