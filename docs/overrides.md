# Overrides

Use overrides when you disagree with the router or want to compare models.

```bash
switchboard route "prompt" --force-model ollama/qwen3:8b
switchboard ask "prompt" --force-model ollama/qwen2.5-coder:7b
switchboard route "prompt" --force-model claude-code
switchboard ask "prompt" --force-model codex
```

Switchboard Core can call installed, authenticated Codex and Claude Code CLIs. It still
does not automate subscription web UIs such as ChatGPT Web.

Embedding-only models such as `ollama/nomic-embed-text` are rejected for route/ask
chat responses. They are reserved for memory/search work.

Sensitive content remains protected by private mode. Unsafe overrides are not implemented:
subscription backends are blocked for sensitive/private prompts.

## Rerun And Escalate

Prompt bodies are not stored by default. Rerun/escalate therefore require the prompt
again unless prompt logging has been explicitly enabled.

```bash
switchboard rerun <request_id> --model ollama/deepseek-r1:8b --prompt "..."
switchboard escalate <request_id> --to manual/claude-web --prompt "..." --show-prompt
```

Escalations record the original request ID, original model, escalated-to model, and
whether the result was recommendation-only.
