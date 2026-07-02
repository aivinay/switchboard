# 2026 Q3 Upgrade Notes

Branch: `feat/2026-q3-upgrade`

Baseline before changes:

- `make install && make check`
- Python: local virtualenv created by `make install`
- Result: 644 tests collected, all passed

## Phase Status

### Phase 1 - P0 correctness fixes

Status: done.

- Fixed route previews for sensitive prompts when Ollama is unavailable. The preview now
  keeps the recommendation on Ollama and states that execution would refuse the request
  rather than send it to Codex or Claude.
- Added regression coverage for the exact sensitive prompt + Ollama unavailable + Claude
  available scenario, including CLI output.
- Added a mypy override for `numpy` internals so the optional router extra remains green
  on Python 3.11 when newer NumPy stubs use Python 3.12 syntax.
- Made `switchboard/config/` the documented canonical config tree, added
  `make sync-config`, and added drift/safe-default tests for the root `config/` copy.

Tests:

- Baseline: 644 collected, passed.
- Phase check: `make check`, 648 collected, passed.

## Manual Follow-Ups

Pending later phases.

## Draft PR Description

### Summary

Implements the planned 2026 Q3 upgrade in phase commits, preserving the deterministic
privacy floor and local-first routing guarantees.

### Phase-by-Phase Changes

- Phase 1: private-mode route preview correctness, Python 3.12/NumPy mypy compatibility,
  and config-tree drift prevention.

### Test Evidence

- Baseline before changes: `make install && make check`, 644 tests collected, all passed.
- Phase 1: `make check`, 648 tests collected, all passed.

### Invariant Checklist

- Deterministic policy still precedes learned components.
- Sensitive content is never routed to subscription/cloud fallback when private mode
  flags it.
- Learned/optional paths fail closed to deterministic routing.
- Telemetry remains metadata-only.
- README benchmark numbers, evaluation claims, and DOI references were not changed.
- New runtime dependencies were not added to the core install.
