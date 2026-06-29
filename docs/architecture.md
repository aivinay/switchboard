# Architecture

Switchboard is a local-first orchestration layer for heterogeneous AI agents. It combines
a local UI/CLI, deterministic policy, learned recall components, runtime grounding,
shared session context, semantic memory, prompt compression, provider adapters,
metadata-only metrics, and evals.

## Core Flow

```text
User / UI / CLI
  |
  v
Session store + current request
  |
  v
Capability detector + deterministic tools
  |
  v
Privacy floor + sensitivity escalation
  |
  v
Rules / learned / LLM router
  |
  v
Context builder + semantic memory + redaction
  |
  v
Compression layer
  |
  v
Adapters: Ollama / Codex CLI / Claude Code
  |
  v
Response sanitizer + metadata-only metrics + evaluation hooks
```

The invariant is simple: deterministic policy runs before learned behavior and remains
the fallback. Learned components can improve recall, but they cannot weaken privacy,
force unavailable backends, skip tool verification, or bypass provider boundaries.

## Layers

- User / UI / CLI: the browser UI and CLI expose one workflow over Auto, Ollama, Codex,
  Claude Code, and deterministic tools.
- Session store: every core `ask` starts by storing the user turn in SQLite and resolving
  a session ID. Successful assistant turns are stored back into the same session.
- Capability detector: regex and heuristic detection identify current-time/date,
  calculation, unit conversion, stock, news, weather, web search, coding, reasoning,
  local, and private signals.
- Tool registry: deterministic tools handle time/date, calculator, unit conversion,
  configured stock quotes, configured news headlines, optional web search, and honest
  live-data fallbacks.
- Privacy floor: private mode blocks sensitive content from non-local backends. Keyword,
  PII, and secret-format detections are final; the learned sensitivity escalator can only
  add protection.
- Router: deterministic rules choose local/coding/reasoning routes. Optional LLM,
  hybrid, and learned router modes classify ambiguous prompts, with rules fallback.
- Context builder: trusted facts, semantic-memory facts, recent conversation, and the
  current request are assembled into tagged blocks and redacted before backend delivery.
- Compression layer: request-level compression handles oversized raw prompts; the
  context-boundary pass compresses recent conversation only and records token metadata.
- Adapters: Codex and Claude Code are local CLI adapters that call user-authenticated
  tools; Ollama runs local models. Switchboard does not automate subscription web UIs or
  resell API access.
- Metrics + Evaluation: backend calls record metadata-only metrics. Mock evals verify
  structure, real smoke evals verify local integration, provider smoke evals check
  configured live-data providers, and the quality benchmark measures routing conditions.

## Two User-Facing Surfaces

Switchboard currently has two related surfaces:

- `switchboard route` previews the same backend decision as the core service without
  calling a model.
- The web UI, bare `switchboard ask`, and `switchboard ask --backend auto` use the fully configured core service.
  This path owns shared sessions, context injection, semantic-memory retrieval,
  Headroom-style context compression, backend adapters, and backend telemetry.

Both surfaces share the same configuration, model catalogue, provider boundaries, local
SQLite database, routing policy, and privacy defaults.

The `/personal/*` API, savings ledger, feedback, rerun, and escalation commands remain
on `PersonalSwitchboardService` for personal accounting and legacy recommendation
workflows; they are not the public CLI `route`/bare `ask` path.

## Boundaries

- `api/`: HTTP request/response handling only.
- `cli.py`: command-line entrypoint; delegates to services.
- `services/switchboard_core.py`: stateful backend orchestration for UI and core CLI asks.
- `services/personal_switchboard.py`: personal route/call, usage, savings, and feedback
  workflow.
- `services/core_factory.py`: builds configured core services from `config/personal.yaml`.
- `services/session_context.py`: shared session prompt assembly and redaction.
- `services/context_compression.py` and `services/compression_layer.py`: token estimates
  and heuristic compression.
- `services/semantic_memory.py`: local semantic-memory embeddings and search.
- `services/tools.py`: deterministic tool orchestration and live-data fallbacks.
- `providers/` and `backends/`: local, cloud, mock, manual, Ollama, Codex, and Claude
  integration boundaries.
- `storage/`: SQLite telemetry, context, feedback, and memory repositories.

## Provider And Data Boundaries

Prompt and response bodies are not stored in backend telemetry by default. Provider keys
are read from environment variables. Session context and memory live in the local SQLite
database. Local embeddings are produced through Ollama; Switchboard does not send memory
embeddings to cloud providers.

Weather remains unsupported as a first-party provider. News and stock grounding are
implemented when configured, and web search is optional. Missing live-data providers
produce honest pass-through facts instead of fabricated current information.
