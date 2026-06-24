# Context, Memory, And Compression

Switchboard's stateful core path gives one conversation a shared working context across
Ollama, Codex, Claude Code, and deterministic tools. The web UI uses this path by
default. In the CLI, use `switchboard ask --backend auto` or a concrete backend such as
`--backend codex` to use the same path.

The older personal route/call surface still powers `switchboard route` and bare
`switchboard ask`. That surface is useful for recommendations, usage accounting, and
local-first calls, but it is not the session-memory path described here.

## Request Flow

1. The user message is written to the SQLite context store.
2. Runtime capabilities are detected: time, date, calculation, unit conversion, stock,
   news, weather, web search, coding, reasoning, local, and private signals.
3. Deterministic tools run before model routing whenever they can ground the answer.
4. The privacy floor runs before subscription backends. Keyword, PII, and secret-format
   matches are final; learned sensitivity can only add protection.
5. The router selects Ollama, Codex, or Claude Code. Learned and LLM routers only fill
   classification gaps and fall back to deterministic rules.
6. The context builder assembles trusted facts, retrieved memory, recent conversation,
   and the current request.
7. The compression layer records token metadata and compresses only the safe parts of
   the prompt.
8. The selected backend receives the assembled prompt. On success, the assistant message
   is stored in the same session for later turns.

## Shared Session Context

Session context is backend-neutral. A turn answered by Codex can be followed by a Claude
Code turn, an Ollama turn, or a deterministic tool answer without losing the recent
conversation.

The context builder emits structured blocks:

```text
<trusted_facts>
- deterministic tool output, live-data grounding, or honesty facts
</trusted_facts>
<long_term_memory>
- retrieved semantic-memory facts
</long_term_memory>
<recent_conversation>
User: ...
Assistant: ...
</recent_conversation>
<current_user_request>
...
</current_user_request>
```

Before any block is shared with a backend, Switchboard redacts recognized secret formats
using the same patterns as the sensitivity classifier. Conversation and memory snippets
are capped and cleaned; the current request keeps newlines and indentation so code, YAML,
and stack traces are not flattened.

## Compression Behavior

Compression has two separate metadata families:

- `compression_*`: request-level compression for oversized raw prompts.
- `context_compression_*`: model-boundary compression after session context has been
  assembled.

The model-boundary pass is structure-aware. If the prompt contains Switchboard's
structured context blocks, only `<recent_conversation>` is compressible. The instruction
preamble, `<trusted_facts>`, `<long_term_memory>`, and `<current_user_request>` survive
that pass byte-for-byte. If the context is still large after history compression,
Switchboard does not delete grounded facts or the user request to hit a budget.

The heuristic compressor is deterministic and dependency-free. For raw text, it keeps a
task header, important code/error snippets when detected, opening context, and the most
recent context. For assembled context, it summarizes history and preserves grounded
truth.

Useful controls:

```yaml
preferences:
  compression_enabled: true
  compression_threshold_tokens: 1000
```

Per-request CLI override:

```bash
switchboard ask --backend auto --no-compression "..."
```

## Semantic Memory

Memory items are local SQLite records. When semantic memory is enabled, Switchboard also
stores local embeddings for those records using Ollama's `nomic-embed-text` model by
default.

```yaml
preferences:
  semantic_memory_enabled: true
  semantic_memory_top_k: 3
  embedding_model: "nomic-embed-text"
```

Add and search memory:

```bash
switchboard memory add \
  --title "Project preference" \
  --content "Prefer local models for private project notes." \
  --project personal

switchboard memory search "private project notes" --project personal
```

When embeddings are available, backend prompts can receive matching memories in the
`<long_term_memory>` block. Memory facts are redacted by the same context-cleaning path
as conversation history.

Direct `memory search` remains useful even if embeddings are unavailable because it uses
SQLite text search. Automatic model-context injection is stricter: it is similarity-gated
and therefore depends on indexed embeddings for reliable injection.

Per-request CLI override:

```bash
switchboard ask --backend auto --memory "Use my project preferences here."
```

## Token And Savings Signals

Switchboard estimates tokens with a lightweight character approximation. The estimates
are used for routing metadata, compression metadata, and the savings ledger; they are not
billed provider usage.

Look for these fields with `--show-metadata`:

- `compression_original_tokens`
- `compression_compressed_tokens`
- `compression_tokens_saved`
- `context_compression_original_tokens`
- `context_compression_compressed_tokens`
- `context_compression_tokens_saved`
- `memory_retrieved_count`
- `semantic_memory_used`
- `context_recent_message_count`

The savings ledger focuses on scarce-model usage and API spend. Compression metadata is
the place to inspect token reduction for a specific model-boundary prompt.

## Privacy Boundaries

- Prompt and response bodies are not stored in telemetry by default.
- Session messages live in the local SQLite context store.
- Feedback context snapshots are opt-in and are skipped for private-mode reroutes and
  learned sensitivity escalations.
- Embeddings are produced locally through Ollama. They are not sent to cloud providers by
  Switchboard.
- Provider API keys are referenced by environment-variable name and are not stored inline
  in configuration.

## Current Boundaries

- Weather has no first-party weather provider yet. Weather prompts either receive an
  honest unsupported/live-data path, route to configured web search, or use Claude Code
  web search when that preference is enabled.
- News and stock grounding are implemented when configured. The shipped local config uses
  Google News RSS and Yahoo finance-style stock grounding.
- Semantic-memory injection depends on indexed embeddings. Text search is available for
  direct memory search even when embeddings are offline.
- Compression is heuristic. It is designed to preserve facts and intent, not to be a
  semantic summarizer with correctness guarantees.
