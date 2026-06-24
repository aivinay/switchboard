# Overrides

Use overrides when you disagree with the router or want to compare models.

```bash
switchboard route "prompt" --force-model ollama/qwen3:8b
switchboard ask "prompt" --force-model ollama/qwen2.5-coder:7b
switchboard ask "prompt" --strict
switchboard route "prompt" --force-model manual/claude-web --show-prompt
switchboard ask "prompt" --force-model manual/codex --show-prompt
```

Manual subscription providers remain recommendation-only. Switchboard never automates
Claude, ChatGPT, or Codex web UIs.

Cloud force-model requests are blocked while `allow_cloud=false` unless the prompt is
non-sensitive and the user passes `--allow-cloud-once`.

Embedding-only models such as `ollama/nomic-embed-text` are rejected for route/ask
chat responses. They are reserved for memory/search work.

Sensitive content remains protected by private mode. Unsafe overrides are not implemented:
cloud and manual providers are blocked for sensitive/private prompts.

For summarisation, strict source grounding is the default. `--strict` makes the local
provider prompt even more explicit, but it is not required for normal source-grounded
summaries.

Add a reason when useful:

```bash
switchboard route "prompt" \
  --force-model manual/claude-web \
  --override-reason "I believe this needs Claude" \
  --show-prompt
```

Override telemetry records the router-selected model, user-forced model, final selected
model, override flag, and override reason.

## Rerun And Escalate

Prompt bodies are not stored by default. Rerun/escalate therefore require the prompt
again unless prompt logging has been explicitly enabled.

```bash
switchboard rerun <request_id> --model ollama/deepseek-r1:8b --prompt "..."
switchboard escalate <request_id> --to manual/claude-web --prompt "..." --show-prompt
```

Escalations record the original request ID, original model, escalated-to model, and
whether the result was recommendation-only.
