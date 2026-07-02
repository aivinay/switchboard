"""Tests for the architecture review fixes: shared embedding cache, learned
sensitivity escalator (escalate-only), dispatcher feedback retraining, and
status-intent consolidation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.models.telemetry import (
    BackendMetricRecord,
    FeedbackExampleRecord,
)
from switchboard.app.services.container import build_container
from switchboard.app.services.learned_router import RouterWeights
from switchboard.app.services.semantic_memory import CachedEmbedder
from switchboard.app.services.sensitivity_escalator import (
    SENSITIVITY_CLASSES,
    LearnedSensitivityEscalator,
)
from switchboard.app.services.status_intents import (
    asks_tool_status,
    asks_web_status,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# CachedEmbedder
# ---------------------------------------------------------------------------


def test_cached_embedder_embeds_each_text_once() -> None:
    calls: list[str] = []

    def embed(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 2.0]

    cached = CachedEmbedder(embed, maxsize=4)
    first = cached.embed("same prompt")
    second = cached.embed("same prompt")
    assert first == second
    assert calls == ["same prompt"]
    assert cached.hits == 1 and cached.misses == 1


def test_cached_embedder_evicts_oldest_and_never_caches_errors() -> None:
    calls: list[str] = []

    def embed(text: str) -> list[float]:
        calls.append(text)
        if text == "boom":
            raise RuntimeError("embedder down")
        return [float(len(text))]

    cached = CachedEmbedder(embed, maxsize=2)
    cached.embed("a")
    cached.embed("b")
    cached.embed("c")  # evicts "a"
    cached.embed("a")  # re-embedded
    assert calls.count("a") == 2

    with pytest.raises(RuntimeError):
        cached.embed("boom")
    with pytest.raises(RuntimeError):
        cached.embed("boom")  # errors are never cached
    assert calls.count("boom") == 2


# ---------------------------------------------------------------------------
# Sensitivity escalator (unit)
# ---------------------------------------------------------------------------

_SENSITIVE_WORDS = {"crying", "marriage", "lump", "owe", "burden", "drinking", "struggling"}


def toy_embed(text: str) -> list[float]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return [2.0 if words & _SENSITIVE_WORDS else 0.0, 1.0]


def escalator_weights() -> RouterWeights:
    return RouterWeights(
        classes=SENSITIVITY_CLASSES,
        embedding_model="toy",
        dim=2,
        mean=[0.0, 0.0],
        std=[1.0, 1.0],
        weights=[[6.0, 0.0], [0.0, 2.0]],  # sensitive <- feature, neutral <- bias
        bias=[0.0, 0.0],
        metadata={},
    )


def escalator(min_confidence: float = 0.7) -> LearnedSensitivityEscalator:
    return LearnedSensitivityEscalator(
        weights=escalator_weights(),
        embed=toy_embed,
        min_confidence=min_confidence,
    )


def test_escalator_escalates_confident_sensitive_phrasings() -> None:
    result = escalator().classify("i've been crying a lot lately")
    assert result.success
    assert result.escalate
    assert result.confidence >= 0.7


def test_escalator_stays_quiet_on_neutral_and_low_confidence() -> None:
    neutral = escalator().classify("write a python function to merge lists")
    assert neutral.success and not neutral.escalate

    strict = escalator(min_confidence=1.0).classify("i've been crying a lot lately")
    assert strict.success and not strict.escalate


def test_escalator_fails_closed_on_embedder_failure() -> None:
    def boom(_: str) -> list[float]:
        raise RuntimeError("down")

    broken = LearnedSensitivityEscalator(weights=escalator_weights(), embed=boom)
    result = broken.classify("i've been crying a lot lately")
    assert not result.success and not result.escalate


def test_escalator_degenerate_embedding_fails_closed() -> None:
    broken = LearnedSensitivityEscalator(
        weights=escalator_weights(),
        embed=lambda _: [1.0, 1.0],
    )
    result = broken.classify("i've been crying a lot lately")
    assert not result.success and not result.escalate
    assert "degenerate_embedding" in (result.error or "")


def test_escalator_from_file_missing_returns_none(tmp_path: Path) -> None:
    assert LearnedSensitivityEscalator.from_file(tmp_path / "missing.json") is None


# ---------------------------------------------------------------------------
# Sensitivity escalator (integration: escalate-only semantics)
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


def make_service(tmp_path: Path, **kwargs) -> tuple[SwitchboardCoreService, dict[str, FakeAdapter]]:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter(
            "claude-code", cost_type=BackendCostType.SUBSCRIPTION
        ),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'archfix.db'}",
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
        **kwargs,
    )
    return service, adapters


def test_escalator_keeps_keyword_missed_private_prompt_local(tmp_path: Path) -> None:
    # "compare" would normally route to Claude; the phrasing has no privacy
    # keywords, but the escalator catches it and privacy policy wins.
    service, adapters = make_service(tmp_path, sensitivity_escalator=escalator())

    response = service.ask(
        "compare my options for telling my boss why i keep crying at work",
        new_session=True,
    )

    assert response.backend == "ollama"
    assert adapters["claude-code"].prompts == []
    record = service.metrics_list(limit=1)[0]
    assert record.metadata.get("sensitivity_escalated") is True


def test_escalator_never_deescalates_keyword_positives(tmp_path: Path) -> None:
    # A keyword-flagged prompt stays local even when the learned model would
    # confidently say "neutral" (here: an escalator that always says neutral).
    class NeutralEscalator(LearnedSensitivityEscalator):
        def classify(self, prompt: str):  # type: ignore[override]
            raise AssertionError("escalator must not be consulted on keyword positives")

    service, adapters = make_service(
        tmp_path,
        sensitivity_escalator=NeutralEscalator(
            weights=escalator_weights(), embed=toy_embed
        ),
    )

    response = service.ask("how do i deal with my depression", new_session=True)

    assert response.backend == "ollama"
    assert adapters["claude-code"].prompts == []
    assert adapters["codex"].prompts == []


def test_no_escalator_means_keyword_behavior_unchanged(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path, sensitivity_escalator=None)

    response = service.ask(
        "compare the tradeoffs of renting versus buying a home", new_session=True
    )

    assert response.backend == "claude-code"


# ---------------------------------------------------------------------------
# Dispatcher feedback retraining
# ---------------------------------------------------------------------------


def _seed_dispatcher_feedback(engine, *, dispatcher_used: bool) -> None:
    from sqlmodel import Session

    with Session(engine) as session:
        session.add(
            BackendMetricRecord(
                request_id="req-1",
                backend="ollama",
                project="personal",
                prompt_char_count=24,
                success=True,
                latency_ms=5,
                cost_type="local",
                metadata_json=json.dumps({"tool_dispatcher_used": dispatcher_used}),
            )
        )
        session.add(
            FeedbackExampleRecord(
                request_id="req-1",
                rating="too-weak",
                detail="bad_answer",
                prompt="what date is the next payday",
                context_text="ctx",
                response_text="wrong grounded answer",
                route_type="tool",
                backend="ollama",
            )
        )
        session.commit()


def _toy_dispatcher_dataset(path: Path) -> None:
    from switchboard.app.services.tool_dispatcher import TOOL_CLASSES

    rows = []
    vocab = {
        "time": "what time is it",
        "date": "what day is today",
        "calculation": "whats 3 plus 5",
        "unit_conversion": "convert feet to miles",
        "stock_price": "tesla stock today",
        "news": "latest headlines",
        "weather": "is it hot outside",
        "none": "tell me a story",
    }
    for label in TOOL_CLASSES:
        for i in range(6):
            rows.append({"prompt": f"{vocab[label]} {'x' * i}", "label": label,
                         "source": "template"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def dispatcher_toy_embed(text: str) -> list[float]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    keys = [
        {"time", "clock"},
        {"day", "today", "date", "payday"},
        {"plus", "divided"},
        {"convert", "feet"},
        {"tesla", "stock"},
        {"headlines", "news"},
        {"hot", "outside", "rain"},
    ]
    return [float(bool(words & k)) for k in keys] + [1.0]


def test_dispatcher_feedback_retrains_and_marks_processed(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    from switchboard.training.feedback_loop import (
        FeedbackExampleStore,
        retrain_dispatcher_with_feedback,
    )

    engine = create_db_engine(f"sqlite:///{tmp_path / 'fb.db'}")
    init_db(engine)
    _seed_dispatcher_feedback(engine, dispatcher_used=True)
    dataset = tmp_path / "dispatcher.jsonl"
    _toy_dispatcher_dataset(dataset)
    weights_path = tmp_path / "dispatcher_weights.json"

    result = retrain_dispatcher_with_feedback(
        engine=engine,
        dataset_path=dataset,
        weights_path=weights_path,
        embed=dispatcher_toy_embed,
        embedding_model="toy",
    )

    assert result["status"] == "deployed", result
    assert weights_path.exists()
    assert FeedbackExampleStore(engine).unprocessed_bad_answer_examples() == []


def test_dispatcher_feedback_skips_non_dispatcher_bad_answers(tmp_path: Path) -> None:
    from switchboard.training.feedback_loop import (
        retrain_dispatcher_with_feedback,
    )

    engine = create_db_engine(f"sqlite:///{tmp_path / 'fb2.db'}")
    init_db(engine)
    _seed_dispatcher_feedback(engine, dispatcher_used=False)

    result = retrain_dispatcher_with_feedback(
        engine=engine,
        dataset_path=tmp_path / "missing.jsonl",
        weights_path=tmp_path / "w.json",
        embed=dispatcher_toy_embed,
    )

    assert result["status"] == "skipped"


# ---------------------------------------------------------------------------
# Status intents (behavior parity)
# ---------------------------------------------------------------------------


def test_status_intents_behavior_parity() -> None:
    assert asks_web_status("is web search configured?")
    assert not asks_web_status("search the web for cats")

    assert asks_tool_status("does switchboard have weather configured?")
    assert asks_tool_status("which providers configured right now")
    assert not asks_tool_status("what's the weather in delhi")
    assert not asks_tool_status("tesla stock price today")


# ---------------------------------------------------------------------------
# Sensitivity dataset hygiene
# ---------------------------------------------------------------------------


def test_sensitivity_dataset_builds_and_golden_never_leaks() -> None:
    from switchboard.training.sensitivity_dataset import (
        sensitivity_examples,
        sensitivity_golden_examples,
    )

    examples = sensitivity_examples()
    labels = {ex.label for ex in examples}
    assert labels == {"sensitive", "neutral"}
    assert len(examples) > 200
    golden = {ex.prompt.lower() for ex in sensitivity_golden_examples()}
    assert not golden & {ex.prompt.lower() for ex in examples}
    # The historic false positive must be a golden NEUTRAL case.
    assert any("personal images" in p for p in golden)
