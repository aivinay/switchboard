# Privacy

Switchboard is local-first by default. It is designed to preserve scarce premium usage
while keeping private work on local routes unless the user explicitly changes
preferences and private mode allows the route.

## Defaults

The starter configuration uses:

```yaml
preferences:
  local_first: true
  allow_cloud: false
  private_mode: true
  require_confirmation_for_scarce_models: true
  cache_routing: true
  cache_answers: false
```

Cloud API providers are disabled by default. Ollama is local and enabled by default.
Provider API keys are read from environment variables when a cloud provider is explicitly
enabled.

## Provider Boundaries

Switchboard separates three concepts:

- Local providers: Ollama and compatible local runtimes.
- Cloud API providers: official API providers, disabled unless configured.
- User-authenticated subscription CLIs: Codex CLI and Claude Code adapters invoked on
  the local machine.

The personal route catalogue also contains manual subscription entries such as
`claude_web` and `chatgpt_web`. Those are recommendation-only. Switchboard does not
automate web UIs, scrape sessions, or bypass provider limits.

Codex CLI and Claude Code are different: the core backend adapters call installed,
user-authenticated local CLI tools. They still obey private mode and provider boundaries;
Switchboard is not reselling access or hiding usage.

## Metadata-Only Logging

Backend telemetry stores routing metadata such as selected backend, route type,
capabilities, tool usage, session ID, context counts, compression counts, success/failure
status, latency, and sanitized errors.

Prompt and response bodies are not logged in telemetry by default. Routing cache entries
store route decisions keyed by a normalized prompt hash and strip prompt-bearing fields
before persistence. Ready-to-paste premium prompts are generated for the current response
and are not stored in the cache by default.

Session messages and memory records are local SQLite data, not telemetry. They are used
to build shared context on the stateful core path.

## Private Mode

When `private_mode=true`, confidential, regulated, and private personal prompts are kept
off cloud and subscription routes. If there is no acceptable local/mock route, the router
blocks or recommends a manual review path rather than routing around the constraint.
Route previews follow the same floor: if a sensitive prompt would require Ollama and the
local runtime is unavailable, the preview says the request would be refused instead of
showing Codex or Claude as a fallback.

Sensitivity does not automatically mean "use the strongest model." Private medical or
regulated content blocks cloud/subscription routes, while task complexity still decides
model strength. Simple private summarisation and extraction can use a local model; complex
private planning can use the strongest allowed local route.

Sensitive reason codes include:

- `PRIVATE_MODE_ENABLED`
- `CLOUD_DISABLED`
- `PROMPT_INJECTION_ATTEMPT`
- `SECURITY_ROUTING_OVERRIDE_ATTEMPT`
- `FINANCIAL_PLANNING_DETECTED`
- `LEGAL_SENSITIVE_CONTENT`
- `MEDICAL_SENSITIVE_CONTENT`
- `PRIVATE_PERSONAL_CONTENT`

Prompts that say things like "ignore private mode", "ignore privacy settings", "mark
this as public", or "use cloud anyway" are treated as routing-override attempts. Those
instructions are recorded as reason codes for auditability; they do not override
`private_mode`, `allow_cloud`, or scarce-model confirmation requirements.

Force-model overrides are constrained by private mode. A local/mock override can be used
for sensitive content, but cloud and subscription overrides are blocked for sensitive or
private prompts. Unsafe override flags are intentionally not implemented.

## Context And Memory Privacy

The context builder redacts recognized secret formats from recent conversation, trusted
facts, semantic-memory facts, and the current request before sharing them with a backend.
The same secret patterns support the sensitivity floor, so routing and redaction do not
drift apart.

Semantic memory uses local SQLite records and local Ollama embeddings by default.
Embeddings are not sent to cloud providers by Switchboard. Memory retrieval can be
disabled with:

```yaml
preferences:
  semantic_memory_enabled: false
```

Compression preserves `<trusted_facts>`, `<long_term_memory>`, and
`<current_user_request>` during the model-boundary pass. It does not delete grounded
facts to satisfy a token threshold.

## Feedback Privacy

Rerun and escalation commands do not recover prompt bodies from telemetry because prompt
bodies are not stored by default. Supply the prompt again with `--prompt` when needed.

Feedback context snapshots are opt-in through `store_feedback_examples`. Even when
enabled, private-mode reroutes and learned sensitivity escalations are not snapshotted.
Those requests can still store a correction label without persisting the assembled
context.

Wrong-model corrections are upserted per request and can be retracted. Retraction and
re-rating to `good` remove pending feedback examples as well as the visible feedback
record. `feedback_auto_retrain: false` keeps opt-in snapshots local but prevents the
automatic threshold retrain path from starting; you can still retrain manually.

## Update Checks

`switchboard version`, `switchboard upgrade --check`, and `switchboard ui` startup may
contact `https://pypi.org/pypi/switchboard-local/json` to see whether a newer public
release exists. The check is synchronous, capped at about one second, cached for 24 hours
in the normal Switchboard config home as `update-check.json`, and skipped in CI.

The request does not include prompts, responses, session data, memory data, API keys, or
telemetry. Like any HTTPS request to PyPI, it still exposes ordinary connection metadata
to PyPI and the network path. Disable it with:

```bash
SWITCHBOARD_UPDATE_CHECK=off
```

or in config:

```yaml
preferences:
  update_check_enabled: false
```

## Hot Local Models

Latency-aware routing never weakens privacy rules. Reusing an already-loaded Ollama model
can avoid a cold start, but only after private mode, provider permissions, task type, and
model suitability are checked. Embedding models are excluded from chat routing, and cloud
or subscription routes remain blocked for sensitive content unless preferences explicitly
allow them and private mode permits the route.

## Tests

Tests must not call real external providers or local model servers unless they are
explicit real-provider or real-smoke evals. Provider integrations go through adapters, and
test fixtures patch real cloud adapters so accidental calls fail fast.
