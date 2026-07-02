# Learned Router

The learned router replaces brittle keyword rules with a tiny embedding
classifier, while keeping every safety-critical decision deterministic.

## Architecture

```
prompt
  -> deterministic policy (runs first, always):
       privacy reroute, tool grounding, live-data, forced backend, follow-up stickiness
  -> learned classifier (only when policy did not already decide):
       embed prompt (preferences.embedding_model) -> softmax over {tool, local, coding, reasoning}
  -> rules fallback (if weights missing, embedder down, or confidence < threshold)
```

The model answers one question only: *what kind of request is this?* It never
decides privacy or availability. A misclassified sensitive prompt is still
caught by the deterministic privacy guard, which runs before the classifier.

## Classes

- `tool` — deterministically answerable (time, math, units, stock, weather,
  news, "who is the president"). The capability detector picks the concrete
  tool; if none fires, this falls back to local.
- `local` — small/simple/private tasks for the local model (includes all
  sensitive/personal topics by construction in the training data).
- `coding` — code, repos, web/app development, algorithms → Codex.
- `reasoning` — architecture, design, tradeoffs, planning, review → Claude Code.

## Training

```bash
# Build synthetic data, embed via Ollama, train, write weights:
switchboard train-router --output config/router_weights.json

# Optionally diversify phrasings with Claude paraphrases (uses subscription):
switchboard train-router --augment --augment-limit 200

# Then enable it:
#   config/personal.yaml -> preferences.router_mode: "learned"
switchboard ask "create a project with a login page" --backend auto --router learned
```

`preferences.embedding_model` is the default embedder for learned routing,
semantic memory, dispatcher training, and sensitivity training. It remains
`nomic-embed-text` for compatibility with shipped weights. To move to a newer
embedder, retrain every learned weight file with the same model:

```bash
switchboard train-router --embedding-model embeddinggemma --output config/router_weights.json
switchboard train-dispatcher --embedding-model embeddinggemma --output config/tool_dispatcher_weights.json
switchboard train-sensitivity --embedding-model embeddinggemma --output config/sensitivity_weights.json

switchboard train-router --embedding-model qwen3-embedding:0.6b --output config/router_weights.json
switchboard train-dispatcher --embedding-model qwen3-embedding:0.6b --output config/tool_dispatcher_weights.json
switchboard train-sensitivity --embedding-model qwen3-embedding:0.6b --output config/sensitivity_weights.json
```

Weights record both `embedding_model` and `dim`. If the configured embedder does
not match the weights metadata, or an embedding vector has the wrong dimension,
the learned component is not used and Switchboard falls back to deterministic
rules.

Training data = template expansion (labeled by the legacy rules) + hand-labeled
golden dogfood cases + optional Claude paraphrases. Every routing bug found
during dogfooding is a golden case the trained model must pass.

## Why this over fine-tuning an LLM

For a 4-class problem, an embedding + softmax-regression head is the right tool:
trains in seconds on CPU, ~50 ms inference (faster than an LLM router call),
pure-Python at inference (no numpy), retrainable nightly from feedback, and its
confidence scores gate the rules fallback. numpy is needed only at training
time (the optional `[router]` extra).

## Which Router Mode Should I Use?

- `rules`: safest and fastest deterministic baseline.
- `learned`: default recommendation for daily use. It keeps deterministic policy first,
  then uses the local embedding classifier for route recall with rules fallback.
- `hybrid`: rules first, then a local LLM judge only when rules classify the prompt as
  unknown.
- `llm`: consults the local LLM judge for every non-policy decision; useful for local
  experiments, but slower than `learned`.

For the LLM judge, `preferences.router_llm_model` defaults to `llama3.2:3b`. Set it to
`hf.co/katanemo/Arch-Router-1.5B.gguf` to use Arch-Router's policy-selection prompt:

```yaml
preferences:
  router_mode: "hybrid"
  router_llm_model: "hf.co/katanemo/Arch-Router-1.5B.gguf"
```

The embedding classifier remains the default because it is faster and cheaper locally.
Arch-Router can be more accurate on ambiguous prompts, with roughly 0.2-1s of local
latency depending on hardware. Any parse failure, timeout, or unavailable router model
falls back to deterministic rules.

## Design invariant

The model replaces classification, never policy. These remain deterministic
forever: sensitive content never reaches a forced subscription backend;
unavailable backends fall back in fixed order; forced selection is never
overridden; deterministic tools ground time/math/unit/stock/news answers.

## Learned tool dispatcher

The regex CapabilityDetector is precise but narrow: measured on CLINC150's
real human phrasings it catches only ~47% of tool-shaped requests (calculator
29%, unit conversion 9%) while false-firing on just 0.4% of non-tool ones.
The learned tool dispatcher recovers the missed recall with the same recipe
as the router — embedding + softmax over {time, date, calculation,
unit_conversion, stock_price, news, weather, none} — under two hard rules:

1. **Regex first.** The dispatcher runs only when the regexes found no tool
   capability and no coding/reasoning/private signal, so the precise fast
   path keeps its behavior.
2. **The tool is the judge.** A prediction counts only if the tool then
   verifies it: the calculator must parse the expression, a ticker must
   resolve. Live classes (news, weather) flow into the existing honest
   live-data policy. Any failure leaves the request exactly as the regexes
   saw it. Learned recall, verified precision.

Train once (CLINC150 + templates, fetched once and cached; ``none`` is a
trained class so the model learns what NOT to dispatch):

```bash
switchboard train-dispatcher   # writes config/tool_dispatcher_weights.json
```

Held-out CLINC150 sweep (hashed bag-of-words lower bound; nomic embeddings
improve both axes): min_confidence 0.8 gives ~60% end-to-end verified recall
at ~1.2% false positives; 0.9 gives ~56% at ~0.5% (regex parity). Default:
0.8 (``preferences.tool_dispatcher_min_confidence``).

## Learned sensitivity escalator

The keyword privacy hints are the floor, not the ceiling. The escalator
embeds the prompt and classifies {sensitive, neutral} to catch phrasings the
keywords miss ("I've been crying a lot lately"). Hard rules: it runs only
when keywords said *not* sensitive, it can only ADD protection (keyword
positives never consult it), and any failure — low confidence, missing
weights, embedder down — leaves the keyword verdict. Train once:

```bash
switchboard train-sensitivity   # writes config/sensitivity_weights.json
```

The golden gate includes the historic false positive ("login page with my
personal images" must NOT escalate) and known keyword misses (must escalate).

## Deterministic safety floor (never learned)

Two protections deliberately stay regex/keyword-based so they work even when
Ollama (and therefore every learned component) is down: the privacy keyword
floor — including physical-health disclosures and secret-format detection
(AWS keys, JWTs, PEM blocks, env-style credentials; single-source patterns in
`app/utils/secret_patterns.py` also drive context redaction) — and the
availability-fallback re-check that blocks sensitive content from
subscription backends. Context compression is structure-aware: only
conversation history is summarized; trusted facts, memory, and the user
request always survive verbatim.

## Shared embeddings

All learned components (router, tool dispatcher, sensitivity escalator,
semantic memory) use the configured local embedder. Router, dispatcher, and
sensitivity calls use a cached classification embedder per request, so a prompt
is embedded once for those classifiers. Semantic memory uses the same configured
model with retrieval-specific document/query prompts.

For `nomic-embed-text`, Switchboard prefixes classifier inputs with
`classification:`, indexed memories with `search_document:`, and memory queries
with `search_query:`. It also sets `num_ctx` explicitly on embedding calls so
longer prompts are not silently limited by the default context. For
`qwen3-embedding:0.6b`, Switchboard prepends a short task instruction before the
text because the model is instruction-aware.

## Feedback also teaches the dispatcher

A thumbs-down "bad answer" on a response the dispatcher grounded becomes a
``none`` training example for the dispatcher; it retrains in the same
background pass as the router, behind its own golden gate.

## External training data (optional)

`switchboard train-router --external` enriches the synthetic dataset with
~1,150 real human utterances, fetched once and cached at
`data/external_router_examples.jsonl`:

| Source | License | Contribution |
| --- | --- | --- |
| CLINC150 (clinc/oos-eval, EMNLP 2019) | CC BY 3.0 | real phrasings of time/date/weather/calculator/conversion/exchange-rate queries -> `tool`; small talk -> `local` |
| databricks-dolly-15k | CC BY-SA 3.0 | human-written QA/summarization/creative instructions -> `local` |
| CodeAlpaca-20k | Apache 2.0 | coding instructions -> `coding` |

External examples are down-weighted (0.4x) relative to templates (1x), golden
cases (2x), and your feedback corrections (3x), so bulk public data informs
the decision boundary without overpowering hand-labeled truth. The training
report prints golden-case accuracy with and without your changes; the
auto-retrain golden gate applies regardless.
