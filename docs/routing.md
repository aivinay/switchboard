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

Legacy `/v1` routes still emit older compatibility reason codes.
