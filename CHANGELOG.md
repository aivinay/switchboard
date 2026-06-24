# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  bundle (Zenodo, doi:10.5281/zenodo.20789935), not this repository.
- FastAPI service, CLI (`switchboard`), and a minimal local web UI.

[Unreleased]: https://github.com/aivinay/switchboard/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aivinay/switchboard/releases/tag/v0.1.0
