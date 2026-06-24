# Local Models

`route` is useful immediately with mock and manual providers because it only recommends.
`ask` becomes truly useful after enabling a real local provider such as Ollama or LM
Studio. Mock answers are demo-only.

Local models are useful, fast, and private, but they can over-infer. For summarisation,
Switchboard adds source-grounding instructions before calling the model: use only the
provided source text, do not invent facts, and prefer fewer accurate bullets over filler.
Quality warnings flag likely padded or speculative summaries without automatically using
premium tools.

## Ollama Setup

Install Ollama, then pull the installed local pack:

```bash
ollama pull llama3.2:3b
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

Stronger machines can also pull heavier variants:

```bash
ollama pull qwen2.5-coder:14b
ollama pull deepseek-r1:14b
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b
```

The heavier model profiles are disabled by default in `config/models.yaml`.

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
switchboard loaded-models
switchboard bench-models
```

`nomic-embed-text` is included for learned routing, semantic memory, and search. It is
never selected for normal chat or ask responses.

Semantic memory uses this embedding model when `semantic_memory_enabled: true`:

```bash
switchboard memory add --title "Preference" --content "Prefer local models for private notes."
switchboard ask --backend auto --memory "Use my saved preference."
```

If the embedding model is unavailable, direct `switchboard memory search` still uses
SQLite text search. Automatic backend-context injection depends on indexed embeddings.

## Routing Roles

- `ollama/llama3.2:3b`: fast summaries, rewrites, classification, extraction.
- `ollama/qwen3:8b`: general local assistant, reasoning, planning, factual Q&A.
- `ollama/qwen2.5-coder:7b`: coding, debugging, code review, code fixing.
- `ollama/deepseek-r1:8b`: complex reasoning, planning, tradeoff analysis.
- `ollama/nomic-embed-text`: embeddings, memory, and search only.

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
switchboard warm ollama/qwen3:8b
switchboard unload ollama/qwen3:8b
```

Recommended warm set:

- `ollama/qwen3:8b` for everyday summaries, planning, and general work.
- `ollama/qwen2.5-coder:7b` when you are actively coding.
- `ollama/deepseek-r1:8b` only when a request needs stronger reasoning.

For a weak or padded local summary, retry locally before spending premium quota:

```bash
switchboard ask "Summarise this email in five bullets: ..." --force-model ollama/qwen3:8b
switchboard feedback <request_id> --rating too-weak --note "Summary over-inferred"
```

Runtime modes:

- `memory_saver`: prefer reusing one already-loaded good-enough model.
- `balanced`: reuse hot models for simple/medium work, but switch for coding and deep reasoning.
- `low_latency`: favor hot good-enough models to avoid cold starts.

Switchboard will not reuse `ollama/nomic-embed-text` for chat responses, and it will not
reuse a general model for coding when the coding model is the better fit.

`bench-models` unloads each Ollama model after benchmarking when
`unload_after_benchmark: true`, so a quick smoke test does not leave every model in RAM.
