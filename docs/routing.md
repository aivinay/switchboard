# Routing

Personal routing is intentionally explicit and testable. The router recommends or calls only models allowed by local personal preferences.

## Default Preferences

- `local_first: true`
- `prefer_free_models: true`
- `allow_cloud: false`
- `require_confirmation_for_scarce_models: true`
- `private_mode: true`
- `router_mode: "learned"` in the shipped personal config, with deterministic rules as
  the fallback when weights are missing or confidence is low.
- `compression_enabled: true`
- `escalation_enabled: false`
- `escalation_confidence_threshold: 0.55`
- `semantic_memory_enabled: true`
- `quota.codex_calls_per_5h:` unset
- `quota.claude_calls_per_week:` unset

## Default Behaviour

- Simple summarisation, classification, and extraction prefer `ollama/gemma4:e4b`,
  with `ollama/llama3.2:3b` kept as the minimal fallback.
- General reasoning and planning prefer `ollama/gemma4:12b`.
- Coding uses `ollama/qwen3.5:9b` when possible.
- Complex reasoning and architecture can use `ollama/gpt-oss:20b` locally first.
- Manual premium tools can still be recommended, but are never called automatically.
- Private or regulated content stays local by default.
- Sensitivity controls where data can go; complexity controls model strength.
- Simple private medical summarisation or question extraction uses a local/mock medium
  route rather than automatically escalating to frontier.
- Complex regulated/private planning or analysis can still use the strongest local/mock
  route.
- Cloud API providers are not called unless enabled and `allow_cloud=true`.
- Manual web subscriptions are recommendation-only. Codex CLI and Claude Code are local
  user-authenticated CLI adapters on the stateful core path.
- Ollama routing checks loaded models and can reuse a hot good-enough model for simple
  or medium work.
- Coding and high-reasoning prompts still switch to specialist models when needed.
- Embedding models are never selected for chat responses.
- The stateful core path adds shared context, semantic memory, and compression after
  routing policy has been applied.

When private mode flags a prompt as sensitive and Ollama is unavailable, `switchboard
route` keeps the recommendation on Ollama and explains that execution would refuse the
request. It must not preview Codex or Claude as a fallback for sensitive content, because
the privacy floor is final.

## Optional Answer-Confidence Cascade

Set `escalation_enabled: true` to let the stateful core path check successful local
Ollama answers before returning them. The check is one short local follow-up prompt that
asks whether the answer is correct, complete, and responsive. If the score is below
`escalation_confidence_threshold`, Switchboard can escalate once to Codex for
coding-flavored prompts or Claude Code otherwise, but only when that backend is available.

Sensitive prompts never escalate to premium backends. If the local confidence check is
low for sensitive content, Switchboard keeps the local answer and appends a note saying
private mode prevented premium escalation. Check failures also fail closed: the local
answer is returned and metadata records that confidence checking was unavailable.

## Optional Quota-Aware Routing

Quota tracking is local and estimate-only. Switchboard records successful premium
backend calls in the existing backend metrics table, then derives rolling-window counts
for Codex over the trailing 5 hours and Claude Code over the trailing 7 days. It does
not scrape provider dashboards, call provider quota APIs, or assume provider limits.

Quota-aware routing is disabled while budgets are unset:

```yaml
quota:
  codex_calls_per_5h:
  claude_calls_per_week:
```

Set either value to a user-declared soft budget to make the core router quota-aware.
The quota layer runs only after forced-backend, privacy, tool-grounding, and deterministic
classification policy. It can move an already-premium preferred route to the other
premium backend when that backend is available, plausible for the route type, and not
constrained. If both premium backends are constrained, Switchboard keeps the request on
Ollama with a quota reason code. It never upgrades a local decision to premium just
because quota is available.

Inspect the estimate:

```bash
switchboard quota
switchboard quota --format json
```

The UI-facing JSON endpoint is `GET /api/quota`.

## CLI Explanations And Reason Codes

The CLI shows friendly `Why` bullets by default, for example:

- Simple summary
- Low complexity
- Local-first enabled
- Cloud disabled
- Premium model avoided
- Private mode enabled
- Sensitive content detected
- Manual premium recommendation only
- No web automation performed
- Cold model switch avoided
- Specialist model is worth loading

Use `switchboard route ... --debug` or `--show-reasons` to inspect raw internal reason
codes.

## Internal Reason Codes

Examples:

- `PERSONAL_SIMPLE_TASK_ROUTED_TO_FREE_LOCAL_MODEL`
- `PERSONAL_CODING_LOCAL_MODEL_PREFERRED`
- `PERSONAL_CLOUD_DISABLED_PREMIUM_RECOMMENDATION_ONLY`
- `PERSONAL_SCARCE_MODEL_NOT_CALLED_AUTOMATICALLY`
- `PERSONAL_PRIVATE_MODE_CLOUD_BLOCKED`
- `PERSONAL_SENSITIVE_SIMPLE_TASK_KEPT_LOCAL`
- `HOT_MODEL_REUSED`
- `HOT_MODEL_GOOD_ENOUGH`
- `MODEL_SWITCH_AVOIDED`
- `SPECIALIST_MODEL_SWITCH_JUSTIFIED`
- `OLLAMA_MODEL_ALREADY_LOADED`
- `OLLAMA_MODEL_NOT_LOADED`
- `QUOTA_ALTERNATE_PREMIUM_SELECTED`
- `QUOTA_BOTH_PREMIUM_CONSTRAINED_LOCAL`
- `QUOTA_PREMIUM_CONSTRAINED_LOCAL`

Legacy `/v1` routes still emit older compatibility reason codes.
