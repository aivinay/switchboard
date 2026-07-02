# Context, Memory, And Compression

Switchboard's stateful core path gives one conversation a shared working context across
Ollama, Codex, Claude Code, and deterministic tools. The web UI uses this path by
default. In the CLI, bare `switchboard ask`, `switchboard ask --backend auto`, and
concrete backends such as `--backend codex` use the same path.

`switchboard route` previews the same backend decision without calling a model.

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

Switchboard decides which brain should answer. Compression only shrinks what
that brain reads.

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
  compression_engine: "heuristic"
```

`compression_engine` defaults to `heuristic` and keeps the core install dependency-free.
To try the optional Headroom adapter:

```bash
pip install "switchboard-local[headroom]"
```

```yaml
preferences:
  compression_enabled: true
  compression_engine: "headroom"
```

The Headroom adapter is replace-by-choice, not default behavior. It sends only the
`<recent_conversation>` block to the `headroom-ai` `compress(messages)` surface. The
instruction preamble, `<trusted_facts>`, `<long_term_memory>`, and
`<current_user_request>` are spliced back in byte-for-byte. If `headroom-ai` is not
installed, cannot download a model, returns an unsupported shape, or raises at runtime,
Switchboard logs the failure once, records `compression_engine: "heuristic"` plus a
`headroom_fallback_reason`, and uses the built-in heuristic for that request.

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
- `compression_engine`
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
- The default compressor is heuristic. Optional Headroom compression is a history-only
  adapter with the same protected-block contract, not a license to delete grounded
  facts to fit a budget.
