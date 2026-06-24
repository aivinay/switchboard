# Contributing to Switchboard

Thanks for your interest in improving Switchboard! Contributions of all kinds
are welcome — bug reports, docs, tests, and features.

## Development setup

Switchboard targets **Python 3.11+**.

```bash
git clone https://github.com/aivinay/switchboard.git
cd switchboard
make install              # creates .venv and installs the package with dev extras
make check                # lint + type-check + full test suite
```

`make install` is equivalent to `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.

To exercise the real backends (optional), install [Ollama](https://ollama.com)
plus the `codex` and `claude` CLIs and pull `llama3.2:3b`, `qwen3:8b`, and
`nomic-embed-text`. Without them, the full test suite still passes against mock
adapters.

## Before you open a pull request

```bash
make lint          # ruff
make typecheck     # mypy
make test          # pytest (the suite must stay green)
```

- Keep changes focused; one logical change per PR.
- Add or update tests for any behavior change. Switchboard is privacy- and
  routing-critical, so new defects should become permanent regression tests
  (see the adversarial dogfooding harness in `scripts/`).
- Run `make check` and make sure it passes before pushing.
- Follow the existing code style (ruff-enforced, 100-col lines).

## Configuration files (two copies)

`switchboard/config/` is the canonical, packaged copy of the default config
files (it is what ships in the wheel). The root `config/` is a byte-identical
mirror used for source checkouts and Docker runs. When you change a default
config file, apply the same edit to **both** copies so they stay in sync.

## Design invariant (please preserve it)

Deterministic policy **always precedes and overrides** the learned components.
Privacy, tool grounding, forced selection, and availability fallback must keep
working when the local model runtime — and therefore every learned component —
is unavailable. PRs that route a sensitive prompt to a non-local backend, or
that let a learned component override the privacy floor, will not be accepted.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened, and your OS
+ Python version. For anything security- or privacy-sensitive, see
[SECURITY.md](SECURITY.md) instead of filing a public issue.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
