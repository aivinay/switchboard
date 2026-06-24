"""Tests for the learned tool dispatcher: classifier, verification gate,
tool natural-phrasing upgrades, and end-to-end integration.

A deterministic toy embedder stands in for nomic-embed-text, so tests need no
Ollama and no numpy at inference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.models.capabilities import Capability
from switchboard.app.services.container import build_container
from switchboard.app.services.deterministic_tools import (
    CalculatorTool,
    UnitConversionTool,
)
from switchboard.app.services.learned_router import RouterWeights
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tool_dispatcher import (
    TOOL_CLASSES,
    LearnedToolDispatcher,
)
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]

# Toy embedding: one signal feature per tool class + bias.
_KEYWORDS: dict[str, set[str]] = {
    "time": {"time", "clock", "oclock", "late"},
    "date": {"date", "day", "today"},
    "calculation": {"divided", "plus", "sum", "calculate", "math"},
    "unit_conversion": {"convert", "ounces", "feet", "grams", "miles"},
    "stock_price": {"stock", "shares", "tesla"},
    "news": {"news", "headlines"},
    "weather": {"hot", "rain", "outside", "weather"},
}


def toy_embed(text: str) -> list[float]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    vector = [float(bool(words & _KEYWORDS[c])) for c in TOOL_CLASSES if c != "none"]
    return [*vector, 1.0]  # bias feature


def handcrafted_weights() -> RouterWeights:
    dim = len(TOOL_CLASSES) - 1 + 1  # one feature per tool class + bias
    weights = [[0.0] * dim for _ in TOOL_CLASSES]
    for i, _ in enumerate(TOOL_CLASSES[:-1]):
        weights[i][i] = 6.0
    weights[-1][-1] = 2.0  # "none" prior on the bias feature
    return RouterWeights(
        classes=TOOL_CLASSES,
        embedding_model="toy",
        dim=dim,
        mean=[0.0] * dim,
        std=[1.0] * dim,
        weights=weights,
        bias=[0.0] * len(TOOL_CLASSES),
        metadata={},
    )


def dispatcher(min_confidence: float = 0.5) -> LearnedToolDispatcher:
    return LearnedToolDispatcher(
        weights=handcrafted_weights(),
        embed=toy_embed,
        min_confidence=min_confidence,
    )


# ---------------------------------------------------------------------------
# Classifier unit tests
# ---------------------------------------------------------------------------


def test_dispatcher_classifies_tool_intents() -> None:
    cases = {
        "what is 87 divided by 4": ("calculation", Capability.CALCULATION),
        "help me convert feet into miles": ("unit_conversion", Capability.UNIT_CONVERSION),
        "is it hot outside": ("weather", Capability.WEATHER),
        "how is tesla stock doing": ("stock_price", Capability.STOCK_PRICE),
        "any big headlines": ("news", Capability.LATEST_INFO),
    }
    for prompt, (tool_class, capability) in cases.items():
        result = dispatcher().classify(prompt)
        assert result.success, f"{prompt!r}: {result.error}"
        assert result.tool_class == tool_class
        assert result.capability == capability


def test_dispatcher_none_class_is_a_clean_no() -> None:
    result = dispatcher().classify("tell me about your favorite philosopher")
    assert not result.success
    assert result.tool_class == "none"
    assert result.error is None  # "none" is a prediction, not a failure


def test_dispatcher_low_confidence_fails_closed() -> None:
    result = dispatcher(min_confidence=0.99).classify("what is 87 divided by 4")
    assert not result.success
    assert "low confidence" in (result.error or "")


def test_dispatcher_embedder_failure_fails_closed() -> None:
    def boom(_: str) -> list[float]:
        raise RuntimeError("ollama down")

    d = LearnedToolDispatcher(weights=handcrafted_weights(), embed=boom)
    result = d.classify("anything")
    assert not result.success
    assert "ollama down" in (result.error or "")


def test_dispatcher_from_file_missing_weights_returns_none(tmp_path: Path) -> None:
    assert LearnedToolDispatcher.from_file(tmp_path / "missing.json") is None
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(handcrafted_weights().to_dict()), encoding="utf-8")
    loaded = LearnedToolDispatcher.from_file(path, embed=toy_embed)
    assert loaded is not None
    assert loaded.classify("what is 87 divided by 4").tool_class == "calculation"


# ---------------------------------------------------------------------------
# Tool natural-phrasing upgrades (what the verification gate relies on)
# ---------------------------------------------------------------------------


def test_calculator_parses_spoken_operators() -> None:
    calc = CalculatorTool()
    cases = {
        "what is 87 divided by 4": "21.75",
        "what is the sum of 3 plus 5": "8",
        "what is 20 times 20 times 30": "12,000",
        "what is 9 minus 12": "-3",
        "what is 2 to the power of 10": "1,024",
    }
    for prompt, expected in cases.items():
        result = calc.answer(prompt)
        assert result.success, f"{prompt!r}: {result.error}"
        assert result.metadata["calculator_result"] == expected, prompt


def test_calculator_still_fails_closed_on_prose() -> None:
    result = CalculatorTool().answer("calculate my chances of promotion")
    assert not result.success


def test_unit_conversion_answers_number_free_rate_questions() -> None:
    converter = UnitConversionTool()
    result = converter.answer("how do you convert ounces to grams")
    assert result.success, result.error
    assert "28.349523" in result.answer

    result = converter.answer("help me convert feet into miles")
    assert result.success, result.error
    assert "0.000189" in result.answer


def test_unit_conversion_rate_ignores_prose_and_mixed_kinds() -> None:
    converter = UnitConversionTool()
    assert not converter.answer("convert my notes to a summary").success
    assert not converter.answer("convert miles to kilograms").success


# ---------------------------------------------------------------------------
# Integration: second-chance dispatch in SwitchboardCoreService
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    def __init__(self, name: str, *, cost_type: BackendCostType = BackendCostType.LOCAL) -> None:
        self.name = name
        self.cost_type = cost_type
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.prompts.append(request.prompt)
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} answered",
            latency_ms=5,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


def make_service(
    tmp_path: Path,
    *,
    tool_dispatcher: LearnedToolDispatcher | None = None,
) -> tuple[SwitchboardCoreService, dict[str, FakeAdapter]]:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'dispatcher.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    service = SwitchboardCoreService(
        registry=BackendRegistry(dict(adapters)),
        metrics=container.backend_metrics_repository,
        container=container,
        tool_dispatcher=tool_dispatcher,
    )
    return service, adapters


def test_dispatcher_grounds_regex_missed_calculation(tmp_path: Path) -> None:
    service, adapters = make_service(tmp_path, tool_dispatcher=dispatcher())

    response = service.ask("what is 87 divided by 4", new_session=True)

    assert response.success
    assert response.backend == "ollama"
    assert "21.75" in adapters["ollama"].prompts[-1]
    record = service.metrics_list(limit=1)[0]
    assert record.metadata.get("tool_dispatcher_used") is True
    assert record.metadata.get("grounded_by_tool") is True


def test_dispatcher_verification_failure_leaves_flow_untouched(tmp_path: Path) -> None:
    # The classifier proposes "calculation" but the calculator cannot parse
    # anything: the prediction must be discarded, not half-applied.
    service, adapters = make_service(tmp_path, tool_dispatcher=dispatcher())

    response = service.ask("calculate my chances of promotion", new_session=True)

    assert response.success
    record = service.metrics_list(limit=1)[0]
    assert record.metadata.get("tool_dispatcher_class") == "calculation"
    assert not record.metadata.get("tool_dispatcher_used")
    assert not record.metadata.get("grounded_by_tool")
    assert "Calculated locally" not in adapters[response.backend].prompts[-1]


def test_dispatcher_skips_when_regex_already_detected(tmp_path: Path) -> None:
    # Regex catches this phrasing itself; the dispatcher must stay out.
    service, _ = make_service(tmp_path, tool_dispatcher=dispatcher())

    response = service.ask("what is 15% of 80", new_session=True)

    assert response.success
    record = service.metrics_list(limit=1)[0]
    assert record.metadata.get("grounded_by_tool") is True
    assert "tool_dispatcher_class" not in record.metadata


def test_dispatcher_blocked_for_coding_prompts(tmp_path: Path) -> None:
    # A coding prompt mentioning a tool-ish word ("sum") must stay a coding
    # task, never a calculator answer.
    service, _ = make_service(tmp_path, tool_dispatcher=dispatcher())

    response = service.ask(
        "write a python function to sum two numbers",
        new_session=True,
    )

    assert response.backend == "codex"
    record = service.metrics_list(limit=1)[0]
    assert "tool_dispatcher_class" not in record.metadata
    assert not record.metadata.get("grounded_by_tool")


def test_dispatcher_live_class_uses_honest_live_data_policy(tmp_path: Path) -> None:
    # Weather has no backing tool; adopting the capability must flow into the
    # existing honest live-data handling (local + anti-fabrication), not a
    # fabricated premium answer.
    service, adapters = make_service(tmp_path, tool_dispatcher=dispatcher())

    response = service.ask("is it hot outside", new_session=True)

    assert response.backend == "ollama"
    assert "Do not invent specific" in adapters["ollama"].prompts[-1]
    record = service.metrics_list(limit=1)[0]
    assert record.metadata.get("tool_dispatcher_class") == "weather"


def test_no_dispatcher_means_no_behavior_change(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, tool_dispatcher=None)

    response = service.ask("what is 87 divided by 4", new_session=True)

    assert response.success
    record = service.metrics_list(limit=1)[0]
    assert "tool_dispatcher_class" not in record.metadata


# ---------------------------------------------------------------------------
# Dataset + trainer
# ---------------------------------------------------------------------------


def test_dispatcher_dataset_builds_with_canned_clinc(tmp_path: Path) -> None:
    from switchboard.training.tool_dispatcher_dataset import (
        build_dispatcher_dataset,
        dispatcher_golden_examples,
        load_or_build_dispatcher_dataset,
    )

    fixture = {
        "train": [
            ["what time is it in tokyo right now", "time"],
            ["whats twelve * twelve", "calculator"],
            ["will it rain this weekend", "weather"],
            ["how many euros is fifty bucks", "exchange_rate"],  # excluded
            ["book me a flight to denver", "book_flight"],  # -> none
        ],
        "val": [],
        "test": [],
    }

    examples = build_dispatcher_dataset(fetch_json=lambda url: fixture)
    labels = {ex.prompt: ex.label for ex in examples}
    assert labels["what time is it in tokyo right now"] == "time"
    assert labels["will it rain this weekend"] == "weather"
    assert labels["book me a flight to denver"] == "none"
    assert "how many euros is fifty bucks" not in labels
    # Templates contribute per-tool labels for classes CLINC lacks.
    assert {"stock_price", "news"} <= set(labels.values())
    # Golden gate prompts never leak into training data.
    golden_prompts = {ex.prompt for ex in dispatcher_golden_examples()}
    assert not golden_prompts & set(labels)

    cache = tmp_path / "dispatcher.jsonl"
    first = load_or_build_dispatcher_dataset(cache, fetch_json=lambda url: fixture)
    assert cache.exists()
    second = load_or_build_dispatcher_dataset(
        cache, fetch_json=lambda url: (_ for _ in ()).throw(AssertionError("no fetch"))
    )
    assert [(e.prompt, e.label) for e in first] == [(e.prompt, e.label) for e in second]


def test_trainer_learns_dispatcher_classes() -> None:
    import pytest

    pytest.importorskip("numpy")
    from switchboard.training.router_dataset import RouterExample
    from switchboard.training.tool_dispatcher_dataset import (
        dispatcher_golden_examples,
    )
    from switchboard.training.train_router import train

    # Synthetic toy dataset spanning all classes in the toy embedding space.
    examples = []
    phrasings = {
        "time": ["what time is it", "is it late", "check the clock"],
        "date": ["what day is today", "whats the date", "which day is it"],
        "calculation": ["whats 3 plus 5", "87 divided by 4", "do the math"],
        "unit_conversion": ["convert feet to miles", "ounces to grams", "convert units"],
        "stock_price": ["tesla stock today", "how are my shares", "stock check"],
        "news": ["latest news", "morning headlines", "news roundup"],
        "weather": ["is it hot outside", "will it rain", "weather report"],
        "none": ["tell me a story", "who was napoleon", "i feel great"],
    }
    for label, prompts in phrasings.items():
        for repeat in range(4):
            for prompt in prompts:
                examples.append(
                    RouterExample(prompt=f"{prompt} {repeat * 'x'}", label=label,
                                  source="template")
                )

    weights, report = train(
        examples,
        embed=toy_embed,
        embedding_model="toy",
        classes=TOOL_CLASSES,
        golden=[
            ex
            for ex in dispatcher_golden_examples()
            if ex.label in {"calculation", "weather", "news", "stock_price"}
        ],
        epochs=200,
    )
    assert report.holdout_accuracy >= 0.8
    trained = LearnedToolDispatcher(weights=weights, embed=toy_embed, min_confidence=0.0)
    assert trained.classify("is it hot outside").tool_class == "weather"
    assert trained.classify("87 divided by 4").tool_class == "calculation"
