"""Tests for the learned embedding router and its integration.

A deterministic toy embedder (bag-of-keywords) stands in for nomic-embed-text,
so tests need no Ollama and no numpy at inference. The trainer test uses numpy
when available and is skipped otherwise.
"""

from __future__ import annotations

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
from switchboard.app.services.container import build_container
from switchboard.app.services.learned_router import (
    ROUTE_TYPES,
    LearnedRouter,
    RouterWeights,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.training.router_dataset import build_dataset, golden_examples

ROOT = Path(__file__).resolve().parents[1]

# Toy 6-dim embedding keyed on signal words per class.
_FEATURES = ["time", "stock", "code", "design", "private", "bias"]


def toy_embed(text: str) -> list[float]:
    import re

    words = set(re.findall(r"[a-z]+", text.lower()))

    def has(*targets: str) -> float:
        return float(bool(words & set(targets)))

    return [
        has("time", "date", "weather", "news", "stock", "price", "today"),
        has("stock", "price", "news", "weather"),
        has("code", "python", "java", "login", "website", "app", "function", "build"),
        has("design", "architecture", "compare", "tradeoff", "tradeoffs", "plan", "review"),
        has("depression", "relationship", "personal", "private", "salary"),
        1.0,
    ]


def handcrafted_weights() -> RouterWeights:
    # Diagonal-ish weights mapping each feature to a class. Order of classes:
    # tool, local, coding, reasoning.
    dim = len(_FEATURES)
    weights = [[0.0] * dim for _ in ROUTE_TYPES]
    weights[0][0] = 4.0  # tool <- time/stock/news feature
    weights[2][2] = 4.0  # coding <- code feature
    weights[3][3] = 4.0  # reasoning <- design feature
    weights[1][4] = 4.0  # local <- private feature
    bias = [0.0, 0.5, 0.0, 0.0]  # mild prior toward local for empty signals
    return RouterWeights(
        classes=ROUTE_TYPES,
        embedding_model="toy",
        dim=dim,
        mean=[0.0] * dim,
        std=[1.0] * dim,
        weights=weights,
        bias=bias,
        metadata={},
    )


def router(min_confidence: float = 0.5) -> LearnedRouter:
    return LearnedRouter(
        weights=handcrafted_weights(),
        embed=toy_embed,
        min_confidence=min_confidence,
    )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("what time is it in tokyo", "tool"),
        ("stock price of apple", "tool"),
        ("write a python login page", "coding"),
        ("design a scalable architecture", "reasoning"),
        ("how to get out of depression", "local"),
    ],
)
def test_learned_router_classifies_toy_space(prompt: str, expected: str) -> None:
    result = router().classify(prompt)
    assert result.success, result.error
    assert result.route_type == expected
    assert 0.0 <= result.confidence <= 1.0
    assert result.probabilities is not None


def test_learned_router_low_confidence_reports_failure() -> None:
    result = router(min_confidence=0.99).classify("hello there")
    assert not result.success
    assert "low confidence" in (result.error or "")


def test_learned_router_handles_embedder_failure() -> None:
    def boom(_: str) -> list[float]:
        raise RuntimeError("ollama down")

    r = LearnedRouter(weights=handcrafted_weights(), embed=boom)
    result = r.classify("anything")
    assert not result.success
    assert "ollama down" in (result.error or "")


def test_learned_router_dim_mismatch_fails() -> None:
    r = LearnedRouter(weights=handcrafted_weights(), embed=lambda _: [1.0, 2.0])
    result = r.classify("anything")
    assert not result.success
    assert "does not match" in (result.error or "")


def test_weights_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "w.json"
    import json

    path.write_text(json.dumps(handcrafted_weights().to_dict()), encoding="utf-8")
    loaded = RouterWeights.from_file(path)
    assert loaded is not None
    assert loaded.classes == ROUTE_TYPES
    assert loaded.dim == len(_FEATURES)
    assert LearnedRouter.from_file(tmp_path / "missing.json", embed=toy_embed) is None


def test_router_from_file_rejects_embedding_model_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "w.json"
    import json

    path.write_text(json.dumps(handcrafted_weights().to_dict()), encoding="utf-8")

    assert (
        LearnedRouter.from_file(
            path,
            embed=toy_embed,
            expected_embedding_model="embeddinggemma",
        )
        is None
    )


# ---------------------------------------------------------------------------
# Integration: learned mode in SwitchboardCoreService, policy preserved
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


def make_service(tmp_path: Path, **kwargs) -> tuple[SwitchboardCoreService, dict]:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'learned.db'}",
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
        router_mode="learned",
        learned_router=router(),
        **kwargs,
    )
    return service, adapters


def test_learned_mode_routes_coding_to_codex(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    decision = service.route(SwitchboardRequest(request_id="r", prompt="write python code"))
    assert decision.backend == "codex"
    assert "learned router" in decision.routing_reason.lower()


def test_learned_mode_routes_reasoning_to_claude(tmp_path: Path) -> None:
    service, _ = make_service(tmp_path)
    decision = service.route(
        SwitchboardRequest(request_id="r", prompt="design the system architecture")
    )
    assert decision.backend == "claude-code"


def test_learned_mode_privacy_policy_overrides_classifier(tmp_path: Path) -> None:
    # A sensitive prompt must stay local even though the deterministic privacy
    # check runs before the learned classifier.
    service, adapters = make_service(tmp_path)
    response = service.ask("how to get out of depression", new_session=True)
    assert response.backend == "ollama"
    assert adapters["claude-code"].prompts == []


def test_learned_mode_falls_back_to_rules_when_embedder_down(tmp_path: Path) -> None:
    def boom(_: str) -> list[float]:
        raise RuntimeError("offline")

    down_router = LearnedRouter(weights=handcrafted_weights(), embed=boom)
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'fb.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    service = SwitchboardCoreService(
        registry=BackendRegistry(adapters),
        metrics=container.backend_metrics_repository,
        container=container,
        router_mode="learned",
        learned_router=down_router,
    )
    req = SwitchboardRequest(request_id="r", prompt="debug this failing pytest")
    decision = service.route(req)
    assert decision.backend == "codex"  # rules still catch "debug"
    assert "used rules" in decision.routing_reason.lower()


# ---------------------------------------------------------------------------
# Trainer (numpy) — learns the toy mapping and nails the golden cases
# ---------------------------------------------------------------------------


def test_trainer_learns_toy_mapping_and_golden_cases() -> None:
    pytest.importorskip("numpy")
    from switchboard.training.train_router import train

    dataset = build_dataset()
    weights, report = train(dataset, embed=toy_embed, embedding_model="toy", epochs=200)

    # The toy embedder is intentionally coarse, but the trainer should still
    # achieve strong hold-out accuracy and high golden-case accuracy.
    assert report.holdout_accuracy >= 0.7
    assert report.golden_accuracy >= 0.75

    trained = LearnedRouter(weights=weights, embed=toy_embed, min_confidence=0.0)
    correct = sum(
        1 for ex in golden_examples() if trained.classify(ex.prompt).route_type == ex.label
    )
    assert correct >= int(0.75 * len(golden_examples()))
