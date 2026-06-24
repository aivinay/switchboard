"""Training data for the learned tool dispatcher.

Classes (see app/services/tool_dispatcher.py): time, date, calculation,
unit_conversion, stock_price, news, weather, none.

Sources:

- CLINC150 (CC BY 3.0): real crowdsourced phrasings of the five tool-shaped
  intents, plus a broad sample of the other ~140 intents as ``none``
  negatives — these teach the model what NOT to dispatch ("cook time",
  "date on my pay stub", ...).
- Router templates: the synthetic tool prompts already labeled by the regex
  detector (which catches 100% of its own templates) give per-tool labels for
  stock and news, which CLINC lacks; local/coding/reasoning templates add
  ``none`` negatives.
- A hand-labeled golden set built from measured detector misses and
  near-miss false positives; used as the deploy gate, never for training.

Labeling rules mirror the verification gate at inference: ambiguous intents
with no backing tool (exchange_rate, timezone) are excluded entirely rather
than guessed.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from switchboard.app.models.capabilities import Capability
from switchboard.app.services.tool_dispatcher import (
    CAPABILITY_BY_TOOL_CLASS,
)
from switchboard.training.external_datasets import (
    _CLINC_URL,
    _http_get_json,
    _usable,
)
from switchboard.training.router_dataset import RouterExample, build_dataset

# CLINC150 intent -> dispatcher class.
CLINC_INTENT_TOOL_CLASSES: dict[str, str] = {
    "time": "time",
    "date": "date",
    "weather": "weather",
    "calculator": "calculation",
    "measurement_conversion": "unit_conversion",
}

# Tool-adjacent intents with no backing tool: excluded from BOTH the positive
# classes and the ``none`` negatives, so the model is not taught a confusing
# boundary it cannot verify.
EXCLUDED_INTENTS = frozenset({"exchange_rate", "timezone", "oos"})

TOOL_CLASS_BY_CAPABILITY: dict[Capability, str] = {
    capability: tool_class
    for tool_class, capability in CAPABILITY_BY_TOOL_CLASS.items()
}


def clinc_dispatcher_examples(
    *,
    fetch_json: Callable[[str], dict] | None = None,
    per_intent: int = 150,
    per_none_intent: int = 6,
) -> list[RouterExample]:
    fetch = fetch_json or _http_get_json
    payload = fetch(_CLINC_URL)
    examples: list[RouterExample] = []
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        for utterance, intent in payload.get(split, []):
            if intent in EXCLUDED_INTENTS or not _usable(utterance):
                continue
            label = CLINC_INTENT_TOOL_CLASSES.get(intent)
            cap = per_intent if label is not None else per_none_intent
            if counts.get(intent, 0) >= cap:
                continue
            counts[intent] = counts.get(intent, 0) + 1
            examples.append(
                RouterExample(
                    prompt=utterance.strip(),
                    label=label or "none",
                    source="external:clinc150",
                )
            )
    return examples


def template_dispatcher_examples(
    *,
    none_cap: int = 250,
    seed: int = 42,
) -> list[RouterExample]:
    """Relabel the router templates with per-tool classes via the regex
    detector (precise on its own templates); non-tool templates become
    ``none`` negatives."""
    from switchboard.app.services.capabilities import CapabilityDetector

    detector = CapabilityDetector()
    tool_examples: list[RouterExample] = []
    none_pool: list[RouterExample] = []
    for example in build_dataset():
        if example.label == "tool":
            detection = detector.detect(example.prompt)
            for capability in detection.capabilities:
                tool_class = TOOL_CLASS_BY_CAPABILITY.get(capability)
                if tool_class is not None:
                    tool_examples.append(
                        RouterExample(
                            prompt=example.prompt,
                            label=tool_class,
                            source=example.source,
                        )
                    )
                    break
            # Templates the regex cannot place are skipped, not guessed.
        else:
            none_pool.append(
                RouterExample(prompt=example.prompt, label="none", source=example.source)
            )
    random.Random(seed).shuffle(none_pool)
    return tool_examples + none_pool[:none_cap]


def dispatcher_golden_examples() -> list[RouterExample]:
    """Hand-labeled gate cases: measured regex misses (must dispatch) and
    near-miss distractors (must NOT dispatch)."""
    positives = [
        ("what is the sum of 3 plus 5", "calculation"),
        ("what is 87 divided by 4", "calculation"),
        ("what is 20 times 20 times 30", "calculation"),
        ("if i win 200000 how do i split it 7 ways", "calculation"),
        ("help me convert feet into miles", "unit_conversion"),
        ("how do you convert ounces to grams", "unit_conversion"),
        ("how can i change inches into meters", "unit_conversion"),
        ("is it six o clock yet", "time"),
        ("when is it right now", "time"),
        ("i would like to know the time", "time"),
        ("what day of the month is it", "date"),
        ("what day it today", "date"),
        ("date please", "date"),
        ("is it hot outside", "weather"),
        ("is it going to rain tonight", "weather"),
        ("how is it outside", "weather"),
        ("how is tesla stock doing", "stock_price"),
        ("did apple shares go up today", "stock_price"),
        ("any big headlines this morning", "news"),
        ("what's going on in the world today", "news"),
    ]
    negatives = [
        "what's the prep time for a garden salad",
        "when is it time for a change in tires",
        "what's the date on my last pay stub",
        "what is the date of the next holiday",
        "repeat what the weather will be like",
        "write a python script that prints the current date",
        "why is the sky blue",
        "tell me a joke about mathematicians",
        "how do interest rates affect stock markets in general",
        "i had a long day and just want to talk",
    ]
    return [
        RouterExample(prompt=prompt, label=label, source="golden")
        for prompt, label in positives
    ] + [
        RouterExample(prompt=prompt, label="none", source="golden")
        for prompt in negatives
    ]


def build_dispatcher_dataset(
    *,
    fetch_json: Callable[[str], dict] | None = None,
    seed: int = 42,
) -> list[RouterExample]:
    """CLINC + templates, deduplicated (golden prompts excluded from training
    so the gate stays honest), deterministically shuffled."""
    examples = clinc_dispatcher_examples(fetch_json=fetch_json)
    examples += template_dispatcher_examples(seed=seed)
    golden_prompts = {ex.prompt.lower() for ex in dispatcher_golden_examples()}
    seen: set[str] = set()
    unique: list[RouterExample] = []
    for example in examples:
        key = example.prompt.lower()
        if key in seen or key in golden_prompts:
            continue
        seen.add(key)
        unique.append(example)
    random.Random(seed).shuffle(unique)
    return unique


def load_or_build_dispatcher_dataset(
    cache_path: str | Path,
    *,
    fetch_json: Callable[[str], dict] | None = None,
) -> list[RouterExample]:
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
    examples = build_dispatcher_dataset(fetch_json=fetch_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {"prompt": example.prompt, "label": example.label, "source": example.source}
                )
                + "\n"
            )
    return examples
