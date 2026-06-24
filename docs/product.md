# Product

Switchboard is a local-first AI router for individual power users. It is meant for one
person with a laptop, local models, optional API keys, subscription CLI agents, and
limited premium-model quota.

## Current Scope

- Local web UI and CLI over Auto, Ollama, Codex CLI, Claude Code, and deterministic tools.
- Personal `/personal/route` endpoint that recommends a route without calling a model.
- Personal `/personal/ask` endpoint that calls only allowed mock/local/cloud routes.
- Stateful core `/api/chat` and `switchboard ask --backend auto` path with shared
  sessions across backend switches.
- Deterministic privacy floor for private, regulated, PII, and secret-format prompts.
- Learned router, tool dispatcher, and sensitivity escalator with deterministic fallback.
- Context builder with trusted facts, recent conversation, long-term memory, and current
  request blocks.
- Headroom-style heuristic compression with token-savings metadata.
- Local semantic memory with Ollama embeddings and SQLite storage.
- Keyless stock/news grounding when configured, plus optional web search.
- SQLite usage diary, savings ledger, local feedback, and metadata-only backend metrics.
- Mock evals, real-backend smoke tests, provider smoke tests, and a quality benchmark
  harness.
- Backward-compatible OpenAI-style `/v1/chat/completions` endpoint.

## Product Rules

- Keep simple and private work local by default.
- Do not send sensitive content to subscription or cloud backends when private mode is on.
- Do not bypass provider limits.
- Do not scrape or automate subscription web UIs.
- Invoke Codex and Claude Code only through local user-authenticated CLI adapters.
- Do not log prompt or response bodies in telemetry by default.
- Preserve grounded facts and the current request during context compression.
- Treat missing live-data providers as an honesty constraint, not permission to guess.
- Require explicit preferences or user action before scarce/cloud usage.

## User Value

- One session can move between local models, coding agents, reasoning agents, and tools
  without manually re-pasting context.
- Local models handle cheap, private, or simple work first.
- Deterministic tools ground current facts before a model has a chance to hallucinate.
- Compression and routing metadata make token savings and scarce-model usage visible.
- Feedback can teach routing behavior locally without changing privacy boundaries.

## Current Boundaries

- Weather is detected but has no first-party weather provider yet.
- News and stock grounding depend on configured providers and may be delayed or
  unavailable.
- Semantic-memory injection depends on indexed local embeddings; direct memory search
  still works with SQLite text search.
- Compression is heuristic and measurable, not a correctness proof.
- Real smoke evals verify integration, while the quality benchmark is the evidence path
  for answer-quality claims.
