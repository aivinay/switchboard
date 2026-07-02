# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `switchboard version`, global `switchboard --version`,
  `switchboard upgrade [--check]`, and the opt-out daily PyPI update check.

### Fixed
- Replaced the stale source-checkout `__version__` fallback with
  `pyproject.toml` discovery.

## [0.3.0] - 2026-07-02

### Added
- Added the 2026 Ollama model pack, hardware-aware `switchboard models
  --recommend`, and explicit local role mappings for general, coding, and
  reasoning routes.
- Added a sub-12 GiB floor tier for `switchboard models --recommend` so tiny
  machines get `llama3.2:3b` instead of heavier 16 GiB recommendations.
- Added embedding-model-aware learned routing, dispatcher, sensitivity, and
  semantic-memory embedding calls, including task prefixes and fail-closed
  weight metadata checks.
- Added disabled-by-default local answer-confidence checks with one-hop premium
  escalation for low-confidence non-sensitive Ollama answers.
- Added first-class Arch-Router LLM judge support via `router_llm_model`, with
  policy-format prompting and fail-closed parsing.
- Added optional `headroom-ai` compression support behind the `headroom` extra
  and `preferences.compression_engine: "headroom"`, while keeping the
  dependency-free heuristic as the default.
- Added a local, estimate-only premium quota ledger with user-declared soft
  budgets, quota-aware premium routing, `switchboard quota`, and `/api/quota`.
- Added the dependency-free web UI upgrade: dynamic backend picker, routing
  chips, private-mode indicator, quota meters, and a savings dashboard fed by
  recorded metrics.

### Changed
- Made the packaged `switchboard/config/` tree canonical, with `make
  sync-config` and drift tests keeping the root `config/` copy aligned for
  development and Docker mounts.
- Updated `switchboard doctor` to point at local model recommendations and
  report the GLM 4.7 Flash Ollama version requirement.

### Removed
- Removed legacy Ollama chat profiles replaced by the 2026 local model pack:
  `ollama/qwen3:8b`, `ollama/qwen2.5-coder:7b`,
  `ollama/deepseek-r1:8b`, `ollama/gemma3:12b`,
  `ollama/qwen2.5-coder:14b`, `ollama/deepseek-r1:14b`, and
  `ollama/mistral-small3.2:24b`. Existing role mappings that reference these
  profiles fall back to defaults; migrate with
  `switchboard models --recommend --apply` or remap
  `preferences.local_model_roles`.

### Fixed
- Made `switchboard route` previews honor private mode when Ollama is
  unavailable: sensitive prompts now preview as local-only/refused instead of
  recommending a subscription fallback.
- Targeted mypy at Python 3.12 syntax so NumPy 2.1+ stubs using PEP 695 parse
  correctly, and annotated pre-existing training variables that this exposes.
- Corrected Headroom installation docs to use the `switchboard-local` package
  name.
- Corrected the real-backend smoke expectation for deterministic time-tool
  prompts, which are grounded and then formatted locally through Ollama.

## [0.2.3] - 2026-07-01

### Changed
- Removed public personal contact details from package, citation, security, and
  conduct metadata.
- Run the container image as an unprivileged user and restrict CI token
  permissions to read-only repository contents.
- Bind dev and Docker Compose servers to localhost by default, keep private
  files out of Docker build contexts, run Compose with a read-only root
  filesystem, and keep HTTP-triggered subscription CLI backends disabled unless
  explicitly enabled.
- Keep full feedback-example snapshots opt-in in the starter config.

### Fixed
- Hardened live-data fetchers to use HTTP(S) client calls with status checks and
  defused XML parsing for RSS feeds.

## [0.2.2] - 2026-06-29

### Fixed
- Made `switchboard eval-real-providers` honor the packaged
  `personal.yaml` Yahoo Finance default, while keeping empty preferences as
  env-fallback and explicit `none` as disabled.
- Built real-provider grounding evals through the configured core-service
  factory so their live-data tool behavior matches normal CLI requests.
- Nudged stock quote answers to include the finance source and delayed-data
  status when a deterministic finance tool grounds the response.

## [0.2.1] - 2026-06-29

### Fixed
- Corrected `switchboard doctor` and `switchboard backends` live-data provider
  status so configured Yahoo Finance, Google News RSS, direct web search, and
  Claude Code WebSearch fallback paths are reported accurately.
- Matched provider status reporting to runtime env-fallback semantics for empty
  `finance_provider` and `news_provider` preferences, while preserving explicit
  `none` as disabled.

## [0.2.0] - 2026-06-29

### Changed
- Unified the public CLI routing path: `switchboard route`, bare
  `switchboard ask`, and `switchboard ask --backend auto` now use the same
  Switchboard Core routing behavior, with `route` acting as a no-model-call
  preview.
- Removed legacy personal-router-only flags from public `route` and `ask`
  commands so unsupported options fail clearly instead of being ignored.
- Rejected manual catalogue IDs such as `manual/codex` on the callable core
  CLI path; users should select callable backends such as `codex`,
  `claude-code`, or `ollama`.

### Fixed
- Preserved forced backend/model choices in `route` next-step guidance,
  including the `--force-model ollama` backend alias.
- Updated README framing and Zenodo DOI references to the current v2 record.

## [0.1.1] - 2026-06-24

### Fixed
- Ship the web UI static assets (`switchboard/app/static/`) in the wheel so `switchboard ui` works from a PyPI install. They were omitted from `package-data` in 0.1.0, so the server crashed on startup with `Directory '.../app/static' does not exist`.

## [0.1.0] - 2026-06-24

First public release.

### Added
- Privacy-first hybrid routing across local Ollama models and subscription CLI
  agents (Codex, Claude Code), with deterministic policy that always precedes
  and overrides three small locally-trained classifiers (backend router, tool
  dispatcher, sensitivity escalator).
- Deterministic private mode and secret-format detection shared with context
  redaction.
- Structure-aware, Headroom-inspired heuristic context compression (history
  only; grounded facts preserved verbatim).
- Local embedding-based long-term semantic memory (`nomic-embed-text`), with
  SQLite text-search fallback.
- Deterministic tools: time/date, safe-AST calculator, unit conversion, keyless
  live stock quotes and news.
- Evaluation suite: 100-case quality benchmark with a 10-condition ablation
  matrix, local LLM-as-judge, deterministic mock evals, and a real-backend smoke
  suite.
- The multi-run experiment harness, statistical aggregation, and figure
  generation used for the paper are distributed with the paper's reproduction
  bundle (Zenodo, doi:10.5281/zenodo.20836918), not this repository.
- FastAPI service, CLI (`switchboard`), and a minimal local web UI.

[Unreleased]: https://github.com/aivinay/switchboard/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/aivinay/switchboard/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/aivinay/switchboard/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/aivinay/switchboard/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/aivinay/switchboard/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/aivinay/switchboard/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/aivinay/switchboard/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/aivinay/switchboard/releases/tag/v0.1.0
