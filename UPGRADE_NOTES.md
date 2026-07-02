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

### Phase 2 - 2026 local model pack + hardware-aware recommendation

Status: done.

- Added the requested 2026 Ollama model profiles:
  `gemma4:e4b`, `gemma4:12b`, `gemma4:26b`, `gemma4:31b`, `qwen3.5:9b`,
  `qwen3.6:27b`, `glm-4.7-flash`, `qwen3-coder:30b`, `gpt-oss:20b`,
  `embeddinggemma`, and `qwen3-embedding:0.6b`.
- Kept `llama3.2:3b` as the minimal local chat fallback and
  `nomic-embed-text` as the compatibility embedding default.
- Added `preferences.local_model_roles` and wired personal routing to prefer
  configured local role mappings before deterministic fallbacks.
- Added `switchboard models --recommend` with Linux/macOS RAM detection,
  hardware tiering, pull-command output, and `--apply`/`--yes` config rewrite.
- Updated `switchboard doctor` to point at `models --recommend` and note that
  `glm-4.7-flash` requires Ollama 0.14.3 or newer.

Tests:

- Phase check: `make check`, 653 collected, passed.

### Phase 3 - embedding upgrade path for learned components

Status: done.

- Kept `preferences.embedding_model` as the first-class embedding preference and
  made training commands use it by default when `--embedding-model` is omitted.
- Added task-specific embedding calls:
  `classification:` for Nomic classifier inputs, `search_document:` for indexed
  memories, and `search_query:` for memory retrieval queries, with explicit
  `num_ctx` for Nomic requests.
- Added instruction-style prompts for `qwen3-embedding:0.6b`.
- Added weight metadata checks so learned router, dispatcher, and sensitivity
  weights fail closed when the configured embedding model differs from the
  recorded training embedder. Runtime vector dimension mismatches also continue
  to fail closed.
- Did not retrain or ship new weights.

Tests:

- Focused Phase 3 tests: embeddings, learned router, tool dispatcher, and
  training-command unavailable paths passed.
- Phase check: `make check`, 657 collected, passed.

### Phase 4 - cascade escalation after local answer-confidence check

Status: done.

- Added `AnswerConfidenceService`, a local yes/no self-check for successful
  Ollama answers.
- Added `preferences.escalation_enabled` (default `false`) and
  `preferences.escalation_confidence_threshold` (default `0.55`).
- Wired the stateful core path so low-confidence local answers can escalate
  once to Codex for coding-flavored prompts or Claude Code otherwise.
- Preserved the privacy floor: sensitive prompts never escalate to premium
  backends; low-confidence sensitive local answers get an honest note instead.
- Check failures fail closed to the local answer with metadata noting the
  confidence check was unavailable.

Tests:

- Focused escalation tests passed.
- Phase check: `make check`, 662 collected, passed.

### Phase 5 - Arch-Router-1.5B as LLM judge option

Status: done.

- Added `preferences.router_llm_model`, preserving `llm_router_model` as a
  compatibility fallback.
- Added first-class Arch-Router detection for
  `hf.co/katanemo/Arch-Router-1.5B.gguf`.
- Arch-Router prompts now define `tool`, `local`, `coding`, and `reasoning` as
  named policies and parse the returned JSON policy selection.
- Generic LLM router calls now request structured JSON output through Ollama's
  `format` schema.
- Parse failures, unknown Arch policies, unavailable models, and timeouts
  continue to fail closed to deterministic rules.

Tests:

- Focused LLM-router tests passed.
- Phase check: `make check`, 667 collected, passed.

### Phase 6 - optional Headroom integration for compression

Status: done.

- Added the optional `headroom` extra (`headroom-ai`) without adding anything
  to the core install.
- Added `preferences.compression_engine`, defaulting to `heuristic`.
- Added `HeadroomLibCompressionLayer`, used only when the preference is
  `"headroom"` and compression is enabled.
- Preserved the protected-block contract: only `<recent_conversation>` is
  passed to Headroom; `<trusted_facts>`, `<long_term_memory>`, and
  `<current_user_request>` are kept byte-identical.
- Added fail-closed fallback for missing imports, unsupported return shapes,
  model-download/runtime errors, and no-savings results. Fallback records
  `headroom_fallback_reason` and uses the existing heuristic compressor.
- Kept small contexts below the configured threshold as no-ops before touching
  the optional Headroom import path.

Tests:

- Focused compression tests: 14 passed.
- Phase check: `make check`, 671 collected, passed.

## Manual Follow-Ups

Recommended 2026 local pulls:

```bash
ollama pull llama3.2:3b
ollama pull gemma4:e4b
ollama pull gemma4:12b
ollama pull qwen3.5:9b
ollama pull gpt-oss:20b
ollama pull embeddinggemma
ollama pull nomic-embed-text
```

Optional heavier profiles:

```bash
ollama pull gemma4:26b
ollama pull gemma4:31b
ollama pull qwen3.6:27b
ollama pull qwen3-coder:30b
ollama pull glm-4.7-flash  # requires Ollama >= 0.14.3
ollama pull qwen3-embedding:0.6b
```

Run `switchboard models --recommend --apply` to update local role mappings after
reviewing the recommendation. Add `--yes` only for noninteractive automation.

Retrain learned weights after changing `preferences.embedding_model`:

```bash
switchboard train-router --embedding-model embeddinggemma --output config/router_weights.json
switchboard train-dispatcher --embedding-model embeddinggemma --output config/tool_dispatcher_weights.json
switchboard train-sensitivity --embedding-model embeddinggemma --output config/sensitivity_weights.json

switchboard train-router --embedding-model qwen3-embedding:0.6b --output config/router_weights.json
switchboard train-dispatcher --embedding-model qwen3-embedding:0.6b --output config/tool_dispatcher_weights.json
switchboard train-sensitivity --embedding-model qwen3-embedding:0.6b --output config/sensitivity_weights.json
```

Enable answer-confidence escalation:

```yaml
preferences:
  escalation_enabled: true
  escalation_confidence_threshold: 0.55
```

Use Arch-Router as the local LLM judge:

```bash
ollama pull hf.co/katanemo/Arch-Router-1.5B.gguf
```

```yaml
preferences:
  router_mode: "hybrid"
  router_llm_model: "hf.co/katanemo/Arch-Router-1.5B.gguf"
```

Try optional Headroom compression:

```bash
pip install "switchboard[headroom]"
```

```yaml
preferences:
  compression_enabled: true
  compression_engine: "headroom"
```

## Draft PR Description

### Summary

Implements the planned 2026 Q3 upgrade in phase commits, preserving the deterministic
privacy floor and local-first routing guarantees.

### Phase-by-Phase Changes

- Phase 1: private-mode route preview correctness, Python 3.12/NumPy mypy compatibility,
  and config-tree drift prevention.
- Phase 2: 2026 local model catalogue, hardware-aware pack recommendation, explicit
  local role mappings, and updated local-model documentation.
- Phase 3: first-class embedding preference for learned components and semantic
  memory, task-specific embedding prompts, and fail-closed weight metadata checks.
- Phase 4: disabled-by-default local answer-confidence check and privacy-preserving
  one-hop escalation for weak local answers.
- Phase 5: Arch-Router policy judge support and structured generic LLM-router output.
- Phase 6: optional Headroom compression adapter with protected-block preservation and
  heuristic fallback.

### Test Evidence

- Baseline before changes: `make install && make check`, 644 tests collected, all passed.
- Phase 1: `make check`, 648 tests collected, all passed.
- Phase 2: `make check`, 653 tests collected, all passed.
- Phase 3: `make check`, 657 tests collected, all passed.
- Phase 4: `make check`, 662 tests collected, all passed.
- Phase 5: `make check`, 667 tests collected, all passed.
- Phase 6: `make check`, 671 tests collected, all passed.

### Invariant Checklist

- Deterministic policy still precedes learned components.
- Sensitive content is never routed to subscription/cloud fallback when private mode
  flags it.
- Learned/optional paths fail closed to deterministic routing.
- Optional Headroom compression only sees recent conversation history and fails closed
  to the dependency-free heuristic.
- Telemetry remains metadata-only.
- README benchmark numbers, evaluation claims, and DOI references were not changed.
- New runtime dependencies were not added to the core install.
