# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/aivinay/switchboard/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/aivinay/switchboard/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/aivinay/switchboard/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/aivinay/switchboard/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/aivinay/switchboard/releases/tag/v0.1.0
