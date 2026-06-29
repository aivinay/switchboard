# CLI UX

The CLI is the primary local workflow for Switchboard. It should make the
recommendation obvious without requiring the user to inspect raw JSON.

After `make install`, either activate the virtual environment:

```bash
source .venv/bin/activate
```

or run commands through `.venv/bin/switchboard`.

## Route

Use `route` when you want a recommendation without calling a model:

```bash
switchboard route "Summarise this email into three bullets."
```

It prints the recommended model/tool, route kind, confidence, estimated cost, premium
usage impact, privacy note, next step, friendly `Why` bullets, and request ID.

The next step line should make the operational state obvious:

- local/mock route is ready and no premium or cloud API is used
- manual premium tool was recommended but not called
- confirmation is required before a scarce route can be called
- cloud is callable only because `allow_cloud=true`
- private mode blocked cloud routing
- ambiguous prompt was kept local and needs more detail if the answer is weak

To see the exact prompt being classified during a route preview:

```bash
switchboard route "Design a database schema and evaluate scaling risk." --show-prompt
```

`--show-prompt` prints the raw prompt after the backend preview; it does not call a
model.

Raw internal reason codes are hidden by default to keep the output readable:

```bash
switchboard route "Summarise this email." --debug
switchboard route "Summarise this email." --show-reasons
```

## Ask

Use bare `ask` when you want Switchboard to route and call through the core backend
path:

```bash
switchboard ask "Rewrite this note to be clearer."
```

With Ollama enabled, `ask` can use real local models for everyday answers. If Ollama is
unavailable, the fallback answer is explicitly labeled as mock/demo output. If the best
route is Codex or Claude Code and the CLI is installed and authenticated, Switchboard
calls that backend directly, still subject to private-mode and availability checks.

For truth/current-info prompts, Phase 2 first uses a specialized tool, then configured web
search, then model pass-through. Normal chat, emotional conversations, coding,
architecture, writing, and summarization do not use web search unless the user explicitly
asks to search. The CLI and UI keep showing a friendly model label plus clean answer text
rather than raw tool metadata.

Stock-price prompts use the same pattern when a finance provider is configured:
StockPriceTool resolves the ticker, fetches the latest available quote, and gives the
selected model trusted quote facts. Without a configured finance provider, the original
stock question passes through normally.

Quality guardrails can add a warning:

```bash
switchboard ask "Build a 5-year financial model comparing debt and equity financing."
```

Warnings are heuristic. They are meant to tell the user when a local/mock answer may be
too thin for the task.

## Stateful Ask

Bare `ask` uses the core session path by default: shared context across backend
switches, semantic-memory retrieval, context compression, backend metrics, and
tool-grounded context all run there. `--backend auto` is the explicit form of the
same behavior.

```bash
switchboard ask --backend auto --new-session "Remember: keep private notes local."
switchboard ask --backend auto --session <session_id> --memory --show-metadata \
  "What should you remember?"
```

The UI uses this stateful core path by default. The CLI should keep the session ID,
display model, route, and user-facing answer easy to read while hiding raw context blocks
unless `--show-metadata` was requested.

## Usage And Feedback

Usage summarizes request mix, estimated API spend, premium units saved, cache hits, and
feedback:

```bash
switchboard usage
```

Feedback is attached to a request ID:

```bash
switchboard feedback <request_id> --rating good
switchboard feedback <request_id> --rating too-expensive
switchboard feedback <request_id> --rating too-weak --preferred-model ollama/qwen3:8b
```

Feedback is stored locally. Repeated weak local routes can nudge future matching requests
toward a stronger local model. Manual premium preferences remain recommendation-only.

## Setup Checks

```bash
switchboard init
switchboard doctor
switchboard models
switchboard demo
```

`init` writes a safe starter `config/personal.yaml` when one does not exist. `doctor`
checks config loading, database reachability, local model server reachability when
enabled, environment variables for enabled cloud providers, optional web/finance provider
status, and privacy defaults. `doctor` and `backends` report providers as configured or
not configured without printing API keys.

Optional real-provider smoke checks stay explicit:

```bash
switchboard eval-real-providers
```

Missing Brave, yfinance, or Alpha Vantage configuration is reported as `NOT_VERIFIED`,
not as a setup failure.

`route` is useful immediately because it only recommends. `ask` is most useful after
`switchboard doctor` and `switchboard models` show that Ollama is enabled and reachable.
