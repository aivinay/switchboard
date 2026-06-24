"""Public-dataset importers for router training.

Real human phrasings complement the synthetic templates, especially for the
``tool`` class where natural language varies most. Sources (all free,
keyless, fetched once and cached as JSONL):

- CLINC150 (clinc/oos-eval, EMNLP 2019, CC BY 3.0): 150-intent dataset of
  real crowdsourced utterances. Utility intents (time, date, weather,
  calculator, measurement conversion, exchange rate) map to ``tool``;
  small-talk intents map to ``local``.
- databricks-dolly-15k (CC BY-SA 3.0): human-written instructions with
  category labels. QA/summarization/creative map to ``local``.
- CodeAlpaca-20k (Apache 2.0): coding instructions -> ``coding``.

Attribution and license notes live in docs/learned_router.md. Imported
examples carry source="external:<name>" so they can be filtered or weighted
independently of templates and feedback.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections.abc import Callable
from pathlib import Path

from switchboard.training.router_dataset import RouterExample

_USER_AGENT = "Switchboard router dataset importer"
_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset={dataset}&config=default&split=train&offset={offset}&length={length}"
)
_CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"

# CLINC150 intent -> router label. Only listed intents are imported.
CLINC_INTENT_LABELS: dict[str, str] = {
    # Deterministically answerable -> tool
    "time": "tool",
    "date": "tool",
    "weather": "tool",
    "calculator": "tool",
    "measurement_conversion": "tool",
    "exchange_rate": "tool",
    # Small talk and simple lookups -> local
    "greeting": "local",
    "goodbye": "local",
    "thank_you": "local",
    "how_old_are_you": "local",
    "what_is_your_name": "local",
    "fun_fact": "local",
    "tell_joke": "local",
    "definition": "local",
    "spelling": "local",
}

# dolly-15k category -> router label (None = skip).
DOLLY_CATEGORY_LABELS: dict[str, str | None] = {
    "summarization": "local",
    "creative_writing": "local",
    "open_qa": "local",
    "general_qa": "local",
    "closed_qa": "local",
    # dolly "brainstorming" is mostly simple listing, not deep reasoning;
    # mislabeling it would dilute the reasoning class, so it is skipped.
    "brainstorming": None,
    "classification": None,
    "information_extraction": None,
}

_MAX_PROMPT_CHARS = 300


def _http_get_json(url: str, timeout_s: int = 30) -> dict:
    request = urllib.request.Request(  # noqa: S310 - fixed dataset hosts.
        url, headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _hf_rows(
    dataset: str,
    *,
    pages: int,
    fetch_json: Callable[[str], dict],
    page_size: int = 100,
) -> list[dict]:
    rows: list[dict] = []
    for page in range(pages):
        url = _HF_ROWS_URL.format(
            dataset=dataset.replace("/", "%2F"),
            offset=page * page_size,
            length=page_size,
        )
        payload = fetch_json(url)
        page_rows = [item.get("row", {}) for item in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
    return rows


def _usable(prompt: str) -> bool:
    return 0 < len(prompt.strip()) <= _MAX_PROMPT_CHARS


def fetch_clinc150(
    *,
    per_intent: int = 60,
    per_intent_local: int = 35,
    fetch_json: Callable[[str], dict] | None = None,
) -> list[RouterExample]:
    fetch = fetch_json or _http_get_json
    payload = fetch(_CLINC_URL)
    examples: list[RouterExample] = []
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        for utterance, intent in payload.get(split, []):
            label = CLINC_INTENT_LABELS.get(intent)
            if label is None or not _usable(utterance):
                continue
            cap = per_intent_local if label == "local" else per_intent
            if counts.get(intent, 0) >= cap:
                continue
            counts[intent] = counts.get(intent, 0) + 1
            examples.append(
                RouterExample(
                    prompt=utterance.strip(),
                    label=label,
                    source="external:clinc150",
                )
            )
    return examples


def fetch_dolly(
    *,
    pages: int = 10,
    cap_per_label: int = 250,
    fetch_json: Callable[[str], dict] | None = None,
) -> list[RouterExample]:
    fetch = fetch_json or _http_get_json
    rows = _hf_rows("databricks/databricks-dolly-15k", pages=pages, fetch_json=fetch)
    examples: list[RouterExample] = []
    counts: dict[str, int] = {}
    for row in rows:
        label = DOLLY_CATEGORY_LABELS.get(str(row.get("category", "")))
        instruction = str(row.get("instruction", ""))
        # Skip instructions that depend on an attached context passage; the
        # router will never see the passage at inference time.
        if label is None or row.get("context") or not _usable(instruction):
            continue
        if counts.get(label, 0) >= cap_per_label:
            continue
        counts[label] = counts.get(label, 0) + 1
        examples.append(
            RouterExample(prompt=instruction.strip(), label=label, source="external:dolly")
        )
    return examples


def fetch_code_alpaca(
    *,
    pages: int = 5,
    cap: int = 400,
    fetch_json: Callable[[str], dict] | None = None,
) -> list[RouterExample]:
    fetch = fetch_json or _http_get_json
    rows = _hf_rows("sahil2801/CodeAlpaca-20k", pages=pages, fetch_json=fetch)
    examples: list[RouterExample] = []
    for row in rows:
        instruction = str(row.get("instruction", ""))
        if row.get("input") or not _usable(instruction):
            continue
        examples.append(
            RouterExample(
                prompt=instruction.strip(),
                label="coding",
                source="external:code_alpaca",
            )
        )
        if len(examples) >= cap:
            break
    return examples


def fetch_external_examples(
    *,
    fetch_json: Callable[[str], dict] | None = None,
    seed: int = 42,
) -> list[RouterExample]:
    """Fetch and merge all external sources, deduplicated and shuffled
    deterministically. Sources that fail are skipped with a warning so an
    offline machine still trains on whatever is cached/synthetic."""
    examples: list[RouterExample] = []
    for name, fetcher in (
        ("clinc150", fetch_clinc150),
        ("dolly", fetch_dolly),
        ("code_alpaca", fetch_code_alpaca),
    ):
        try:
            examples.extend(fetcher(fetch_json=fetch_json))
        except Exception as exc:
            print(f"  external source {name} skipped: {type(exc).__name__}: {exc}")
    seen: set[str] = set()
    unique: list[RouterExample] = []
    for example in examples:
        key = example.prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    random.Random(seed).shuffle(unique)
    return unique


def load_or_fetch_external(
    cache_path: str | Path,
    *,
    fetch_json: Callable[[str], dict] | None = None,
) -> list[RouterExample]:
    """Load external examples from the JSONL cache, fetching once if absent."""
    path = Path(cache_path)
    if path.exists():
        examples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                examples.append(
                    RouterExample(
                        prompt=row["prompt"],
                        label=row["label"],
                        source=row.get("source", "external"),
                    )
                )
        return examples
    examples = fetch_external_examples(fetch_json=fetch_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "prompt": example.prompt,
                        "label": example.label,
                        "source": example.source,
                    }
                )
                + "\n"
            )
    return examples
