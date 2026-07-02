# Local Development

## Setup

```bash
make install
```

## Run

```bash
make dev
```

Open `http://localhost:8000/docs` for FastAPI's generated API docs.

If `8000` is already in use, run a second dev server on another port:

```bash
make dev PORT=8010
```

## Demo

```bash
source .venv/bin/activate
switchboard demo
bash scripts/demo_personal.sh
```

## Test

```bash
make test
make lint
make typecheck
```

## Configuration

- `.env.example` lists supported environment variables.
- `switchboard/config/` is the canonical packaged config tree.
- `config/personal.yaml` contains the development/Docker-mount copy of local-first user
  preferences and provider toggles.
- `config/models.yaml` contains the development/Docker-mount copy of mock, local, cloud,
  and manual-subscription model profiles.
- SQLite is the default local database.

After editing packaged defaults, refresh the root development copy with:

```bash
make sync-config
```

## Local Models

Ollama is enabled by default in `config/personal.yaml` for this laptop workflow. Pull the
recommended local model pack before expecting real local answers:

```bash
ollama pull llama3.2:3b
ollama pull gemma4:e4b
ollama pull gemma4:12b
ollama pull qwen3.5:9b
ollama pull gpt-oss:20b
ollama pull embeddinggemma
ollama pull nomic-embed-text
```

Use `switchboard models --recommend` for a hardware-aware pack and pull-command list.

The mock provider remains available for tests and fallback. LM Studio is disabled unless
you explicitly enable it in `config/personal.yaml`.

## Cloud and Manual Providers

Cloud APIs are optional and disabled by default. Use environment variables for API keys. Manual subscriptions are recommendation-only and must not automate web UIs.
