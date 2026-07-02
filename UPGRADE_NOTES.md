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
- Removed the legacy Ollama chat profiles replaced by the 2026 pack:
  `ollama/qwen3:8b`, `ollama/qwen2.5-coder:7b`,
  `ollama/deepseek-r1:8b`, `ollama/gemma3:12b`,
  `ollama/qwen2.5-coder:14b`, `ollama/deepseek-r1:14b`, and
  `ollama/mistral-small3.2:24b`. If an existing
  `preferences.local_model_roles` mapping still points at one of these, routing
  falls back to defaults until the mapping is updated.
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

### Phase 7 - quota ledger + quota-aware routing preference

Status: done.

- Added a local, estimate-only quota ledger derived from existing backend
  metrics. It records no prompt or response bodies and counts successful
  premium backend calls only.
- Added top-level `quota.codex_calls_per_5h` and
  `quota.claude_calls_per_week` soft-budget settings, both unset by default.
- Added quota-aware routing after forced-backend, privacy, tool-grounding, and
  deterministic classification policy. It can move an already-premium route to
  the other plausible premium backend when that backend is available and not
  constrained.
- When both premium backends are constrained, Switchboard prefers Ollama and
  records quota reason metadata. Quota never upgrades a local route to premium.
- Added `switchboard quota` and the UI-facing `GET /api/quota` endpoint.

Tests:

- Focused quota/core/UI tests: 129 passed.
- Phase check: `make check`, 679 collected, passed.

### Phase 8 - UI upgrade

Status: done.

- Added UI-only backend status and dashboard endpoints:
  `GET /api/backends/status` and `GET /api/dashboard`.
- Kept `POST /api/chat` compact while enriching streaming and history metadata
  for routing chips.
- Replaced the hardcoded model menu with a dynamic picker that keeps Auto first,
  shows availability dots, and displays hot Ollama models when visible.
- Added assistant routing chips for backend, route type, privacy floor, tool
  grounding, compression, escalation, and quota influence.
- Added a topbar savings dashboard fed by recorded backend metrics, compact
  quota meters fed by `/api/quota`, and a private-mode lock indicator in the
  composer.
- Kept the UI vanilla JS/CSS with no frontend dependencies or CDNs.

Tests:

- Focused UI tests: 34 passed.
- Phase check: `make check`, 681 collected, passed.
- Local UI smoke: `GET /ui`, `/api/backends/status`, `/api/dashboard`, and
  `/api/quota` returned 200 on `127.0.0.1:8765`.

### Phase A - verification findings

Status: done.

- Replaced the ineffective NumPy mypy import override with `python_version =
  "3.12"` so NumPy 2.1+ stubs using PEP 695 syntax parse before import policy
  applies.
- Added the pre-existing train-router type annotations exposed by the new mypy
  target.
- Corrected Headroom installation docs to use the PyPI distribution name
  `switchboard-local[headroom]`.
- Documented the removed legacy Ollama chat profiles and the migration path via
  `switchboard models --recommend --apply` or manual
  `preferences.local_model_roles` remapping.

Tests:

- Phase check: `make check`, 681 collected, passed with NumPy 2.5.0 installed
  on Python 3.14. Python 3.11 was not re-verified locally.

### Phase B - floor tier for `models --recommend`

Status: done.

- Added a `floor` RAM tier below 12 GiB.
- The floor tier maps general, coding, and reasoning roles to
  `ollama/llama3.2:3b` and embeddings to `ollama/embeddinggemma`.
- `switchboard models --recommend` now prints a note that heavier local models
  need more RAM and quota-aware routing matters more on tiny machines.
- Existing 16 GB, 32 GB, and 48 GB+ tiers are otherwise unchanged.

Tests:

- Focused recommendation tests: 7 passed.
- Phase check: `make check`, 683 collected, passed.

### Phase C - real-backend validation

Status: done.

- Exercised the real Ollama, Codex, and Claude Code backends available on this
  machine.
- Confirmed the privacy floor, embedding-model fail-closed path, live
  answer-confidence escalation, configured provider smoke tests, and UI
  telemetry endpoints.
- Fixed one validation-discovered stale real-smoke expectation:
  `real_time_india` now expects the deterministic time tool to route through
  Ollama local formatting instead of Claude Code.
- Reverted all temporary `config/personal.yaml` validation edits and ran
  `make sync-config` after each config flip.

Tests:

- Focused eval tests: `python -m pytest tests/test_evals.py -q`, 23 passed.
- Real smoke after fix: `switchboard eval-real-smoke --fast --timeout 180`,
  11/11 passed.
- Provider smoke: `switchboard eval-real-providers --timeout 180`, 3/5 passed
  and 2/5 not verified because the web provider is not configured.
- Phase check after the validation fix: `make check`, 684 collected, passed.

## Validation

| Step | Commands | Observed result |
| --- | --- | --- |
| Environment survey | `ollama list`; `switchboard doctor`; `switchboard backends`; `switchboard models --recommend` | Ollama is reachable with `embeddinggemma`, `llama3.2:3b`, `nomic-embed-text`, and legacy local models installed. Core backends `ollama`, `codex`, and `claude-code` are available. News is configured through Google News RSS and finance through Yahoo; direct web search is not configured. RAM detected as 32.0 GiB, recommending the 32gb pack: `gemma4:12b`, `qwen3.5:9b`, `gpt-oss:20b`, and `embeddinggemma`. No recommendation pulls were automatic. |
| Embedding pull | `ollama pull embeddinggemma`; `ollama list` | Pulled `embeddinggemma` at 621 MB, under the 1 GB approval threshold, and confirmed it is installed. |
| Default learned routing and semantic memory | `switchboard route "Debug this failing Python pytest traceback" --debug`; `switchboard memory add ... --project validation`; `switchboard memory search ... --project validation` | Default `nomic-embed-text` learned routing classified the coding prompt as `coding` with confidence 1.00 and selected Codex. Semantic memory indexed a validation item and close semantic searches returned the stored item. |
| Embedding preference mismatch | Temporarily set `preferences.embedding_model: "embeddinggemma"`, ran `make sync-config`, then `switchboard route "Debug this failing Python pytest traceback" --debug` | Router, tool-dispatcher, and sensitivity weights reported they were trained with `nomic-embed-text`; Switchboard fell closed to deterministic routing with no crash. Preference was reverted to `nomic-embed-text` and synced. |
| Privacy invariant with Ollama up | `switchboard route "my ssn is 123-45-6789, summarize this medical record" --debug`; `switchboard ask --backend auto --show-metadata --timeout 120 ...` | Route preview selected Ollama with the private-mode sensitive-content reason. The live answer came from `ollama/llama3.2:3b`, cost type `local`, with no premium escalation. |
| Live answer-confidence escalation | Temporarily set `escalation_enabled: true`; ran `switchboard route ... --debug`; `switchboard ask --backend auto --show-metadata --timeout 180 "Please answer carefully: how should projection lag interact with deletion semantics in a stateful notes app?"` | The prompt routed locally first. The confidence check scored the local answer at 0.20 below the 0.55 threshold and escalated once to `claude-code` with the documented routing reason. |
| Sensitive low-confidence behavior | Temporarily set `escalation_enabled: true` and `escalation_confidence_threshold: 1.01`; ran `switchboard ask --backend auto --show-metadata --timeout 180 "my ssn is 123-45-6789, summarize this medical record and explain any uncertainty carefully"` | The response stayed on `ollama/llama3.2:3b`, appended the honest local-confidence note, and recorded that private mode blocked premium escalation. Escalation settings were reverted and synced. |
| Arch-Router optional validation | Not run | Skipped pending explicit approval because `hf.co/katanemo/Arch-Router-1.5B.gguf` is over 1 GB. |
| Real smoke suite | `switchboard eval-real-smoke --fast --timeout 180` | First run found one stale eval expectation: `real_time_india` expected Claude Code but deterministic time-tool formatting correctly used Ollama. Fixed the fixture with a regression test, then reran: 11/11 passed, 0 failed, 0 timed out, 0 skipped, 0 not verified. |
| Real provider suite | `switchboard eval-real-providers --timeout 180` | 3/5 passed with configured providers; `provider_web_brave` and `provider_web_grounding` were not verified because the direct web provider is not configured. |
| Headroom optional validation | Not run | Skipped pending explicit approval because it installs the optional `switchboard-local[headroom]` extra into the venv. |
| UI validation | `switchboard ui --host 127.0.0.1 --port 8765`; `curl http://127.0.0.1:8765/ui`; `curl http://127.0.0.1:8765/api/backends/status`; `curl -X POST http://127.0.0.1:8765/api/chat ...`; `curl http://127.0.0.1:8765/api/dashboard` | `/ui` rendered over GET. The dynamic picker reported Auto and Ollama available, Codex and Claude gated by HTTP CLI opt-in, and hot local models populated. A forced Ollama UI chat returned `OK.` from backend `ollama`. Dashboard totals populated from real backend metrics. Server was stopped after validation. |

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
This is also the migration path for configs that still reference removed legacy
Ollama profiles; alternatively remap `preferences.local_model_roles` by hand.

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
pip install "switchboard-local[headroom]"
```

```yaml
preferences:
  compression_enabled: true
  compression_engine: "headroom"
```

Enable quota-aware routing with your own soft budgets:

```yaml
quota:
  codex_calls_per_5h: 40
  claude_calls_per_week: 200
```

Inspect quota windows:

```bash
switchboard quota
switchboard quota --format json
```

Run the local web UI:

```bash
switchboard ui
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
- Phase 7: local premium quota ledger, soft-budget-aware premium rerouting, and
  CLI/UI quota surfaces.
- Phase 8: dependency-free web UI dashboard, dynamic picker, route chips,
  quota meters, and private-mode affordance.

### Test Evidence

- Baseline before changes: `make install && make check`, 644 tests collected, all passed.
- Phase 1: `make check`, 648 tests collected, all passed.
- Phase 2: `make check`, 653 tests collected, all passed.
- Phase 3: `make check`, 657 tests collected, all passed.
- Phase 4: `make check`, 662 tests collected, all passed.
- Phase 5: `make check`, 667 tests collected, all passed.
- Phase 6: `make check`, 671 tests collected, all passed.
- Phase 7: `make check`, 679 tests collected, all passed.
- Phase 8: `make check`, 681 tests collected, all passed.

### Invariant Checklist

- Deterministic policy still precedes learned components.
- Sensitive content is never routed to subscription/cloud fallback when private mode
  flags it.
- Learned/optional paths fail closed to deterministic routing.
- Optional Headroom compression only sees recent conversation history and fails closed
  to the dependency-free heuristic.
- Quota-aware routing only affects already-premium preferred routes after privacy and
  tool policy, and never upgrades local routes to premium.
- UI dashboard and chips are derived from recorded metadata and do not expose prompt or
  response bodies through metrics endpoints.
- Telemetry remains metadata-only.
- README benchmark numbers, evaluation claims, and DOI references were not changed.
- New runtime dependencies were not added to the core install.
