# Switchboard Docs

Start here when you want the implementation-level view behind the README.

## Core System

- [Architecture](architecture.md): service layers, request flow, and boundaries.
- [Context, memory, and compression](context-memory-compression.md): shared sessions,
  semantic memory, token metadata, and compression behavior.
- [Routing](routing.md): local-first model selection and reason codes.
- [Learned router](learned_router.md): embedding classifiers, tool dispatch, and
  sensitivity escalation.
- [Privacy](privacy.md): local-first defaults, private mode, telemetry, and provider
  boundaries.

## Operation

- [Usage and feedback](usage.md): CLI workflows, metadata, quality warnings, and local
  feedback.
- [Local models](local-models.md): Ollama setup, model roles, and hot-model routing.
- [Performance](performance.md): loaded-model reuse and local runtime modes.
- [Savings](savings.md): scarce-model accounting and reports.
- [Overrides](overrides.md): force-model and safety rules.
- [Local development](local-development.md): setup and development commands.
- [Demo](demo.md): short end-to-end demo script.

## Product

- [Product](product.md): current scope and product rules.
- [UX](ux.md): user-facing behavior and interaction principles.
