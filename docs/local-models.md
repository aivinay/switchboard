# Local Models

`route` is useful immediately because it previews the backend decision without calling a
model. `ask` becomes truly useful after enabling a real backend such as Ollama, LM
Studio, Codex, or Claude Code. Mock answers are demo-only.

Local models are useful, fast, and private, but they can over-infer. For summarisation,
Switchboard adds source-grounding instructions before calling the model: use only the
provided source text, do not invent facts, and prefer fewer accurate bullets over filler.
Quality warnings flag likely padded or speculative summaries without automatically using
premium tools.

## Ollama Setup

Install Ollama, then ask Switchboard for the pack that fits this machine:

```bash
switchboard models --recommend
```

The default 2026 pack is:

```bash
ollama pull llama3.2:3b
ollama pull gemma4:e4b
ollama pull gemma4:12b
ollama pull qwen3.5:9b
ollama pull gpt-oss:20b
ollama pull embeddinggemma
ollama pull nomic-embed-text
```

On machines below roughly 12 GiB of RAM, `switchboard models --recommend` uses a
minimal floor tier:

```bash
ollama pull llama3.2:3b
ollama pull embeddinggemma
```

That tier maps general, coding, and reasoning roles to `ollama/llama3.2:3b`. Heavier
local models need more RAM, so quota-aware routing to Codex or Claude Code matters more
on very small machines.

Stronger machines can also pull heavier variants:

```bash
ollama pull gemma4:26b
ollama pull gemma4:31b
ollama pull qwen3.6:27b
ollama pull qwen3-coder:30b
ollama pull glm-4.7-flash
```

The heavier model profiles are disabled by default in `config/models.yaml`. `glm-4.7-flash`
requires Ollama 0.14.3 or newer.

## Ollama Default

`config/personal.yaml` enables Ollama by default for this laptop setup:

```yaml
providers:
  ollama:
    type: "local"
    base_url: "http://localhost:11434"
    enabled: true
```

Then verify:

```bash
switchboard doctor
switchboard models
switchboard models --recommend
switchboard loaded-models
switchboard bench-models
```

`nomic-embed-text` remains the compatibility embedding default for learned routing,
semantic memory, and search. `embeddinggemma` and `qwen3-embedding:0.6b` are available as
the 2026 embedding upgrade path. Embedding models are never selected for normal chat or
ask responses.

Semantic memory uses this embedding model when `semantic_memory_enabled: true`:

```bash
switchboard memory add --title "Preference" --content "Prefer local models for private notes."
switchboard ask --backend auto --memory "Use my saved preference."
```

If the embedding model is unavailable, direct `switchboard memory search` still uses
SQLite text search. Automatic backend-context injection depends on indexed embeddings.

## Routing Roles

- `ollama/llama3.2:3b`: minimal fallback for existing installs.
- `ollama/gemma4:e4b`: fast summaries, rewrites, classification, extraction.
- `ollama/gemma4:12b`: general local assistant, reasoning, planning, factual Q&A.
- `ollama/qwen3.5:9b`: coding, debugging, code review, code fixing.
- `ollama/gpt-oss:20b`: complex reasoning, planning, tradeoff analysis.
- `ollama/glm-4.7-flash`: optional fast reasoning/coding profile; requires Ollama
  0.14.3 or newer.
- `ollama/nomic-embed-text`: embeddings, memory, and search only.
- `ollama/embeddinggemma`: 2026 embedding model for memory and search.
- `ollama/qwen3-embedding:0.6b`: instruction-aware 2026 embedding model.

## Do I Need All Models Running?

No. Ollama can keep models hot in memory, but running everything is wasteful on a
laptop. Switchboard checks `ollama ps` before routing and can reuse a loaded model when
it is good enough for the task.

For a 32 GB MacBook, start with:

```yaml
local_runtime:
  performance_mode: "balanced"
  max_loaded_models: 2
  keep_alive: "10m"
  reuse_hot_model_if_good_enough: true
  model_switch_penalty_ms: 3000
  prefer_hot_model_for_simple_tasks: true
  unload_after_benchmark: true
```

Useful commands:

```bash
switchboard loaded-models
switchboard warm ollama/gemma4:12b
switchboard unload ollama/gemma4:12b
```

Recommended warm set:

- `ollama/gemma4:12b` for everyday summaries, planning, and general work.
- `ollama/qwen3.5:9b` when you are actively coding.
- `ollama/gpt-oss:20b` only when a request needs stronger reasoning.

For a weak or padded local summary, retry locally before spending premium quota:

```bash
switchboard ask "Summarise this email in five bullets: ..." --force-model ollama/gemma4:12b
switchboard feedback <request_id> --rating too-weak --note "Summary over-inferred"
```

Runtime modes:

- `memory_saver`: prefer reusing one already-loaded good-enough model.
- `balanced`: reuse hot models for simple/medium work, but switch for coding and deep reasoning.
- `low_latency`: favor hot good-enough models to avoid cold starts.

Switchboard will not reuse embedding models for chat responses, and it will not reuse a
general model for coding when the coding model is the better fit.

`bench-models` unloads each Ollama model after benchmarking when
`unload_after_benchmark: true`, so a quick smoke test does not leave every model in RAM.
