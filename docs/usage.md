# Usage And Feedback

Usage is local metadata only. Prompt and response bodies are not stored in telemetry by
default.

```bash
switchboard usage
switchboard savings --days 7
switchboard loaded-models
```

The web UI's Savings drawer uses the same local metrics plus `/api/dashboard` and
`/api/quota`; it shows request counts, token-savings estimates, declared soft-budget
windows, and feedback counts without dollar figures.

The web UI sidebar lists chats started in the UI plus CLI sessions you explicitly title.
Search spans all non-deleted sessions, including CLI one-shots. Deleting a session is a
soft delete with a short Undo window; old tombstones are purged on the next UI startup.

## Route And Ask

`switchboard route` previews the backend decision without calling a model:

```bash
switchboard route "refactor the auth module and add tests"
switchboard route "review this architecture"
```

The preview runs the same core pre-routing checks as `ask`, including deterministic
tool grounding when configured, but it does not call Ollama, Codex, or Claude Code.

Bare `switchboard ask` uses the same stateful core path as `switchboard ask --backend auto`.
That path carries context across Ollama, Codex, Claude Code, and tools. The web UI uses
it automatically:

In the web UI, Private chat forces Ollama for every message in the current session and
persists that flag server-side. Turning it off confirms that earlier messages in the
chat may be included as context for premium backends from then on. The UI is
unauthenticated; keep `switchboard ui` on loopback unless you intentionally allow remote
access.

```bash
switchboard ask "Summarise this customer email in three bullets."
switchboard ask --backend auto --new-session "Remember: use local models for private notes."
switchboard ask --backend auto --session <session_id> --memory "What preference did I give you?"
switchboard ask --backend codex --session <session_id> "Apply that preference to this repo task."
```

Use `--show-metadata` to inspect routing, context, memory, compression, and token fields:

```bash
switchboard ask --backend auto --show-metadata "Summarise this long context..."
```

Useful stateful options:

- `--session <id>`: continue a known session.
- `--new-session`: start a fresh session.
- `--backend auto`: use configured routing while staying on the core path.
- `--backend ollama|codex|claude-code`: force a core backend, still subject to private
  mode.
- `--router rules|llm|hybrid|learned`: override router mode for this ask.
- `--no-compression`: disable compression for this ask.
- `--memory`: enable semantic memory for this ask when config would otherwise leave it
  off.

## Verifying Local Answers

Use `switchboard loaded-models` or `ollama ps` to see what is already running locally:

```bash
switchboard loaded-models
ollama ps
```

After `ask`, Switchboard prints backend metadata:

```text
---
Backend: codex
Model: gpt-5.5
Success: True
Cost type: subscription
Request ID: req_...
```

`Backend: ollama` means a local Ollama model produced the answer. `Backend: codex` or
`Backend: claude-code` means the installed, authenticated CLI backend was called.

Use the request ID for feedback, escalation, or support.

## Memory Commands

Memory is local SQLite data. When semantic memory is enabled and the embedding model is
available, `memory add` also indexes an embedding for later context retrieval.

```bash
switchboard memory add \
  --title "Project preference" \
  --content "Prefer local models for private project notes." \
  --project personal

switchboard memory search "private project notes" --project personal
```

If embedding indexing fails, the command reports it and direct `memory search` still uses
SQLite text search. Automatic injection into backend context depends on indexed semantic
matches.

## Quality Warnings

Switchboard checks a few simple format promises on the personal API path, such as requested
bullet counts, JSON, tables, and one-sentence answers. Source-grounded summaries get
extra checks for padded or speculative bullets.

For summaries, local models are instructed to:

- summarise only the provided source text
- avoid invented facts and assumptions
- produce fewer bullets when the source has fewer distinct facts
- say when only X distinct facts were present in the source

Examples:

```bash
switchboard ask "Give me exactly three bullets: ..." --force-model ollama/gemma4:12b
switchboard route "Create a board-level risk analysis" --force-model claude-code
```

The first retry stays local with a stronger Ollama model. The second previews Claude
Code as the backend without calling it.

## Feedback

Feedback helps label routing decisions for later tuning. It does not automatically change
routing to premium tools.

```bash
switchboard feedback <request_id> --rating good
switchboard feedback <request_id> --rating too-weak --preferred-model manual/claude-web
switchboard feedback <request_id> --rating too-weak --note "Summary invented a bullet"
switchboard feedback <request_id> --rating too-expensive --preferred-model ollama/llama3.2:3b
switchboard feedback <request_id> --rating wrong-route --note "This should have been coding"
```

When previous feedback for the same project and task says a local answer was too weak
and names a stronger local model, future route/ask decisions can prefer that local model.
The CLI keeps this simple:

```text
Feedback: previous feedback considered
```

Manual premium preferences remain recommendation-only; Switchboard will not call Claude,
ChatGPT, or Codex web sessions because of feedback.

In the web UI, thumbs-up and thumbs-down are mutually exclusive and reversible. A
"wrong model" correction is stored as the corrected backend, counts once per request,
and can be retracted; retracting or changing the rating to "good" removes any pending
wrong-model training example. `GET /api/feedback/pending` reports the count of
unprocessed wrong-model corrections. Set `feedback_auto_retrain: false` to keep those
corrections pending until you run `switchboard train-router` yourself.

Usage and savings reports show counts for:

- good
- too weak
- too expensive
- wrong route
- most common preferred models

Route history also records runtime metadata for auditability:

- performance mode
- loaded local models seen at routing time
- whether the selected model was already loaded
- whether a cold model switch was avoided
- whether an Ollama cold start was expected
- compression token estimates
- memory retrieval counts on the core path

Future routing work can use this local feedback to tune personal preferences.
