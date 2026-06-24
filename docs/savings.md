# Savings

The savings ledger is designed for a 1-3 week personal experiment: use local models
first, preserve scarce Claude/ChatGPT/Codex quota, then inspect what happened.

Every personal route/ask/rerun/escalate records:

- task type, complexity, sensitivity
- router-selected model
- user-forced model, if any
- final selected model
- route kind
- whether a model was called
- whether a manual recommendation was returned
- premium units saved or spent
- estimated API spend and API cost saved
- baseline model and baseline source
- cache hit and feedback rating
- local Ollama calls, mock fallback calls, cloud calls, and manual recommendations

Defaults live in `config/personal.yaml`:

```yaml
savings:
  default_baseline_model: "manual/claude-web"
  premium_unit_value_usd: null
  assume_premium_for_unknown: false
```

Run:

```bash
switchboard savings --days 7
switchboard savings --days 14
switchboard savings --since 2026-05-31 --format json
```

If the actual route is local/mock and the baseline is manual/cloud, Switchboard counts
one premium unit saved. Manual recommendations do not count as spent unless the user
forces or escalates to that manual model.

The text report includes total requests, local Ollama calls, mock calls, cloud calls,
manual recommendations, premium units saved/spent, top saved task types, top models,
override/escalation counts, cache hits/misses, feedback, and baseline assumptions.

## Token Savings

The savings ledger is about scarce-model usage and API spend. Prompt compression reports
token reduction separately on the stateful core path:

```bash
switchboard ask --backend auto --show-metadata "..."
```

Look for `compression_*` and `context_compression_*` metadata fields. They use
Switchboard's lightweight token estimate and are useful for comparing prompts and
ablation runs; they are not provider billing records.
