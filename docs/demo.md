# Demo

The Switchboard demo is a local laptop workflow over Auto, Codex, Claude Code, Ollama,
deterministic tools, shared session context, semantic memory, compression, and
metadata-only metrics.

## Run

```bash
make install
source .venv/bin/activate
switchboard doctor
switchboard demo
switchboard ui
```

Open `http://127.0.0.1:8080/ui`.

## Demo Flow

| Step | Prompt | Expected |
| --- | --- | --- |
| Auto routing | `Fix this failing Python test` | Auto selects Codex for coding/debugging. |
| Reasoning | `Review this architecture for a model router` | Auto selects Claude Code for design/reasoning. |
| Local/private | `Answer locally: summarize this sentence` | Auto selects Ollama for local/simple work. |
| Deterministic tool | `Time in India` | TimeTool answers without model guessing. |
| Stock grounding | `What is ServiceNow stock trading at?` | Configured finance provider grounds the quote or returns an honest provider fallback. |
| News grounding | `Latest news about OpenAI` | Configured news provider supplies trusted headlines or returns an honest provider fallback. |
| Weather boundary | `Weather in India` | Weather is detected; without a configured weather/search path, Switchboard avoids inventing live weather. |
| Shared context | `Remember: Switchboard routes between Codex, Claude, and Ollama.` Then switch model and ask `What did I ask you to remember?` | The next backend receives recent session context. |
| Semantic memory | Add a memory with `switchboard memory add`, then ask with `--backend auto --memory` | Indexed local memories can be injected as `<long_term_memory>`. |
| Compression metadata | Ask a long-context question with `--backend auto --show-metadata` | Metadata includes request/context compression token estimates and savings. |

## CLI State Demo

```bash
switchboard ask --backend auto --new-session \
  "Remember: prefer local models for private notes."

switchboard ask --backend auto --session <session_id> --memory --show-metadata \
  "What should you remember, and why does it matter?"
```

The first response prints a session ID. Reuse it in the second command.

## What It Proves

- Routing: Auto chooses among Codex, Claude Code, Ollama, and deterministic tools using
  explainable policy.
- Policy boundaries: private mode and missing live-data providers constrain routing
  before a model is asked to guess.
- Runtime grounding: model-backed prompts can receive trusted time/date, finance, news,
  web-search, or honesty facts.
- Shared context: Switchboard owns recent session context across backend switches.
- Semantic memory: local memories can be embedded and retrieved into backend context.
- Compression: long prompts and long sessions record token estimates and savings without
  deleting trusted facts or the current request in the context-boundary pass.
- Telemetry: metrics are hidden from the UI but recorded locally without prompt or
  response bodies by default.
- Auditability: eval reports and metrics show selected backend, route type, latency,
  status, context counts, memory counts, compression counts, and sanitized errors.

## Evidence Commands

```bash
switchboard eval --mock
switchboard eval-real-smoke --fast
switchboard eval-real-smoke --timeout 90 --output real_smoke_results.json
switchboard metrics summary
```

Current verified baseline:

- Tests: 605 passed, 5 skipped
- Mock evals: 64/64 passed
- Fast real smoke evals: 11/11 passed
- Full real smoke evals: 13/13 passed locally in the recorded 90 second run

Real smoke is an integration check, not a full answer-quality benchmark.
