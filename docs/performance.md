# Performance

Switchboard is latency-aware for local Ollama routing. It checks which
models are already loaded, avoids cold starts when a hot model is good enough, and still
switches to a specialist model when quality matters.

## Runtime Settings

Defaults live in `config/personal.yaml`:

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

Modes:

- `memory_saver`: best for tight RAM. Reuse one loaded good-enough model when possible.
- `balanced`: recommended for a 32 GB MacBook. Keep one general model and one specialist hot.
- `low_latency`: optimize for fewer cold starts on simple and medium-complexity tasks.

## Loaded-Model Routing

Routing order:

1. Choose the ideal model for task type, complexity, privacy, and provider preferences.
2. Check currently loaded Ollama models.
3. If the ideal model is already loaded, use it.
4. If a hot local model is good enough for low or medium-complexity work, reuse it.
5. If the request needs coding or high reasoning, switch to the specialist model.

Reason codes make this explainable:

- `OLLAMA_MODEL_ALREADY_LOADED`
- `OLLAMA_MODEL_NOT_LOADED`
- `HOT_MODEL_REUSED`
- `HOT_MODEL_GOOD_ENOUGH`
- `MODEL_SWITCH_AVOIDED`
- `SPECIALIST_MODEL_SWITCH_JUSTIFIED`
- `MEMORY_SAVER_MODE_ACTIVE`
- `BALANCED_RUNTIME_MODE_ACTIVE`
- `LOW_LATENCY_MODE_ACTIVE`

Embedding models are excluded from chat routing even if they are loaded.

## Recommended 32 GB MacBook Setup

Use `balanced`, keep `max_loaded_models: 2`, and warm models only for the work you are
doing:

```bash
switchboard warm ollama/gemma4:12b
switchboard loaded-models
```

Add `ollama/qwen3.5:9b` while coding. Let `ollama/gpt-oss:20b` load only for
hard planning, debugging, or tradeoff analysis.

## Benchmarking

`switchboard bench-models` smoke-tests enabled local/mock models. When
`unload_after_benchmark: true`, it unloads each Ollama model after its benchmark so the
benchmark does not leave your laptop carrying every model in memory.
