"""Tests for the public-dataset importers (no network: canned fixtures)."""

from __future__ import annotations

from pathlib import Path

from switchboard.training.external_datasets import (
    CLINC_INTENT_LABELS,
    DOLLY_CATEGORY_LABELS,
    fetch_clinc150,
    fetch_code_alpaca,
    fetch_dolly,
    fetch_external_examples,
    load_or_fetch_external,
)

CLINC_FIXTURE = {
    "train": [
        ["what time is it in tokyo right now", "time"],
        ["how many euros is fifty bucks", "exchange_rate"],
        ["whats twelve times twelve", "calculator"],
        ["hey there how are ya", "greeting"],
        ["tell me something funny", "tell_joke"],
        ["book me a flight to denver", "book_flight"],  # unmapped -> skipped
    ],
    "val": [["will it rain this weekend", "weather"]],
    "test": [["convert 3 miles to kilometers", "measurement_conversion"]],
}


def clinc_fetch(url: str) -> dict:
    assert "clinc" in url
    return CLINC_FIXTURE


def hf_fetch_factory(rows_by_dataset: dict[str, list[dict]]):
    def fetch(url: str) -> dict:
        for name, rows in rows_by_dataset.items():
            if name in url:
                if "offset=0" in url:
                    return {"rows": [{"row": row} for row in rows]}
                return {"rows": []}
        raise AssertionError(f"unexpected url {url}")

    return fetch


def test_clinc_mapping_and_skip() -> None:
    examples = fetch_clinc150(fetch_json=clinc_fetch)
    labels = {ex.prompt: ex.label for ex in examples}
    assert labels["what time is it in tokyo right now"] == "tool"
    assert labels["how many euros is fifty bucks"] == "tool"
    assert labels["whats twelve times twelve"] == "tool"
    assert labels["will it rain this weekend"] == "tool"
    assert labels["convert 3 miles to kilometers"] == "tool"
    assert labels["hey there how are ya"] == "local"
    assert labels["tell me something funny"] == "local"
    assert "book me a flight to denver" not in labels
    assert all(ex.source == "external:clinc150" for ex in examples)


def test_dolly_mapping_skips_context_and_unmapped() -> None:
    fetch = hf_fetch_factory(
        {
            "dolly": [
                {"instruction": "Why is the sky blue?", "category": "open_qa", "context": ""},
                {
                    "instruction": "Summarize the passage",
                    "category": "summarization",
                    "context": "a very long passage...",  # depends on context -> skip
                },
                {"instruction": "Brainstorm names for a cafe", "category": "brainstorming",
                 "context": ""},
                {"instruction": "Classify these items", "category": "classification",
                 "context": ""},
            ]
        }
    )
    examples = fetch_dolly(fetch_json=fetch, pages=1)
    labels = {ex.prompt: ex.label for ex in examples}
    assert labels == {"Why is the sky blue?": "local"}


def test_code_alpaca_skips_rows_with_input() -> None:
    fetch = hf_fetch_factory(
        {
            "CodeAlpaca": [
                {"instruction": "Write a function to reverse a string", "input": ""},
                {"instruction": "Fix this code", "input": "def broken(): pass"},
            ]
        }
    )
    examples = fetch_code_alpaca(fetch_json=fetch, pages=1)
    assert [ex.prompt for ex in examples] == ["Write a function to reverse a string"]
    assert examples[0].label == "coding"


def test_merge_dedupes_and_survives_source_failure() -> None:
    def flaky_fetch(url: str) -> dict:
        if "clinc" in url:
            return CLINC_FIXTURE
        raise RuntimeError("offline")

    examples = fetch_external_examples(fetch_json=flaky_fetch)
    assert len(examples) == 7  # clinc fixture only; HF sources skipped
    assert len({ex.prompt for ex in examples}) == len(examples)


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = tmp_path / "external.jsonl"
    first = load_or_fetch_external(cache, fetch_json=clinc_fetch)
    assert cache.exists()

    def must_not_fetch(url: str) -> dict:
        raise AssertionError("cache should have been used")

    second = load_or_fetch_external(cache, fetch_json=must_not_fetch)
    assert [(e.prompt, e.label) for e in first] == [(e.prompt, e.label) for e in second]


def test_label_maps_only_use_known_route_types() -> None:
    valid = {"tool", "local", "coding", "reasoning"}
    assert set(CLINC_INTENT_LABELS.values()) <= valid
    assert {v for v in DOLLY_CATEGORY_LABELS.values() if v} <= valid


def test_relabel_toolable_moves_detector_matches_to_tool() -> None:
    from switchboard.training.router_dataset import (
        RouterExample,
        relabel_toolable,
    )

    examples = [
        # Detector recognizes these as groundable -> relabeled to tool.
        RouterExample(prompt="how many ounces is 2 kg to lbs", label="local",
                      source="external:dolly"),
        RouterExample(prompt="what is 15% of 80", label="local", source="template"),
        RouterExample(prompt="latest news about the markets", label="local",
                      source="external:dolly"),
        # Not tool-shaped -> untouched.
        RouterExample(prompt="why is the sky blue", label="local", source="external:dolly"),
        # Golden and feedback are hand truth -> never relabeled.
        RouterExample(prompt="what is 15% of 80", label="local", source="golden"),
        RouterExample(prompt="latest news please", label="local", source="feedback"),
    ]
    relabeled, changed = relabel_toolable(examples)

    labels = [(e.prompt, e.source, e.label) for e in relabeled]
    assert changed == 3
    assert ("what is 15% of 80", "template", "tool") in labels
    assert ("latest news about the markets", "external:dolly", "tool") in labels
    assert ("why is the sky blue", "external:dolly", "local") in labels
    assert ("what is 15% of 80", "golden", "local") in labels
    assert ("latest news please", "feedback", "local") in labels


def test_relabel_toolable_weather_requires_live_intent() -> None:
    """Conversational live phrasings ("gonna rain", "chance of rain") are
    tool questions; bare topic mentions ("how are weather forecasts
    created?") are not and must keep their original label."""
    from switchboard.training.router_dataset import (
        RouterExample,
        relabel_toolable,
    )

    examples = [
        RouterExample(prompt="is it gonna rain in seattle tomorrow?", label="local",
                      source="external:dolly"),
        RouterExample(prompt="chance of rain this weekend in delhi", label="local",
                      source="external:dolly"),
        RouterExample(prompt="How are weather forecasts created?", label="local",
                      source="external:dolly"),
    ]
    relabeled, changed = relabel_toolable(examples)

    labels = {e.prompt: e.label for e in relabeled}
    assert changed == 2
    assert labels["is it gonna rain in seattle tomorrow?"] == "tool"
    assert labels["chance of rain this weekend in delhi"] == "tool"
    assert labels["How are weather forecasts created?"] == "local"
