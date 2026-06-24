# Security & Privacy Policy

## Reporting a vulnerability

Please **do not** open a public issue for security or privacy vulnerabilities.
Instead, email **ai.vinaygupta@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce, and
- any suggested remediation.

You can expect an acknowledgement within a few days. Please allow reasonable
time for a fix before public disclosure.

## Privacy posture (what Switchboard promises)

Switchboard is local-first and privacy-aware by design:

- **Deterministic privacy floor first.** A keyword/PII/secret-format check runs
  before any non-local routing; a positive verdict is final and cannot be
  overridden by a learned component or by prompt wording.
- **Private mode** blocks sensitive prompts from subscription/cloud backends,
  including on availability fallback.
- **Metadata-only telemetry.** Prompt and response bodies are not stored by
  default; telemetry records routing metadata only.
- **Local embeddings & judge.** Semantic memory embeddings and the evaluation
  judge run locally; embeddings never leave the machine.
- **Secret-format detection** shares its patterns with context redaction so the
  routing boundary and the redactor cannot drift apart.

If you find a way to make a sensitive prompt reach a non-local backend, that is
a security issue — please report it privately as above.

## Handling secrets

- Never commit real credentials. Provider API keys are referenced by environment
  variable name in `config/personal.yaml` (e.g. `OPENAI_API_KEY`), never inline.
- `.env` and `*.db` are git-ignored. Do not force-add them.
