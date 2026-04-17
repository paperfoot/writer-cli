# Contributing

Thanks for wanting to make `writer` better.

## Quick start

```bash
git clone https://github.com/paperfoot/writer-cli.git
cd writer
cargo build
cargo test
./target/debug/writer agent-info
```

## Rules

- **Keep the framework contract.** Every command returns a JSON envelope when piped, exits with a semantic code in 0-4, and has a tested `suggestion` on every error. See `AGENTS.md`.
- **No interactive prompts.** Agents cannot press keys. Destructive ops take `--confirm` as a flag.
- **Errors are recovery plans**, not status reports. Every error includes a machine-readable code and a literal instruction the user can follow.
- **`agent-info` must match reality.** If you add a command, add it to the manifest.

## Pull requests

1. Fork and branch from `main`.
2. Keep the change focused. One feature or fix per PR.
3. Update `agent-info` if you touched the command surface.
4. `cargo test` must pass.
5. Open the PR and describe what changed and why.

## Bug reports

Open an issue with:
- Your OS and `writer --version`
- The command you ran
- What you expected
- What happened instead (paste the JSON error envelope if you have one)

## License

By contributing, you agree your code ships under the MIT License in `LICENSE`.
