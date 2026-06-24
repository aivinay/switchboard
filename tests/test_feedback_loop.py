"""Tests for the closed feedback loop: snapshot storage, thumbs-down
disambiguation, threshold trigger, and the golden-accuracy gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.models.telemetry import FeedbackExampleRecord
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.training.feedback_loop import (
    ROUTE_TYPE_BY_BACKEND,
    FeedbackExampleStore,
    maybe_trigger_retraining,
    retrain_with_feedback,
)

ROOT = Path(__file__).resolve().parents[1]


def make_engine(tmp_path: Path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'loop.db'}")
    init_db(engine)
    return engine


def wrong_model_example(prompt: str, corrected: str) -> FeedbackExampleRecord:
    return FeedbackExampleRecord(
        request_id=f"req_{abs(hash(prompt)) % 99999}",
        rating="wrong-route",
        detail="wrong_model",
        corrected_backend=corrected,
        prompt=prompt,
        context_text="<recent_conversation>...</recent_conversation>",
        response_text="a bad answer",
        backend="ollama",
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_counts_and_purge(tmp_path: Path) -> None:
    store = FeedbackExampleStore(make_engine(tmp_path))
    store.add_example(wrong_model_example("write a login page", "codex"))
    store.add_example(
        FeedbackExampleRecord(
            request_id="req_b", rating="too-weak", detail="bad_answer", prompt="x"
        )
    )

    counts = store.counts()
    assert counts["total"] == 2
    assert counts["unprocessed_wrong_model"] == 1  # bad_answer is excluded

    assert store.purge() == 2
    assert store.counts()["total"] == 0


def test_recent_context_roundtrip_and_cap(tmp_path: Path) -> None:
    store = FeedbackExampleStore(make_engine(tmp_path))
    store.save_recent_context("req_1", "context one")
    store.save_recent_context("req_1", "context one updated")
    assert store.get_recent_context("req_1") == "context one updated"
    assert store.get_recent_context("req_missing") == ""


def test_duplicate_feedback_replaces_example_latest_verdict_wins(tmp_path: Path) -> None:
    store = FeedbackExampleStore(make_engine(tmp_path))
    first = wrong_model_example("write a login page", "codex")
    request_id = first.request_id
    store.add_example(first)
    store.mark_processed([store.unprocessed_wrong_model_examples()[0].id])

    second = wrong_model_example("write a login page", "claude-code")
    second.request_id = request_id
    store.add_example(second)

    assert store.counts()["total"] == 1  # replaced, not appended
    examples = store.unprocessed_wrong_model_examples()
    assert len(examples) == 1  # processed flag reset for the new verdict
    assert examples[0].corrected_backend == "claude-code"


def test_mark_processed(tmp_path: Path) -> None:
    store = FeedbackExampleStore(make_engine(tmp_path))
    store.add_example(wrong_model_example("p1", "codex"))
    records = store.unprocessed_wrong_model_examples()
    assert len(records) == 1
    store.mark_processed([records[0].id], gate_failed=False)
    assert store.unprocessed_wrong_model_count() == 0


# ---------------------------------------------------------------------------
# Trigger threshold
# ---------------------------------------------------------------------------


def test_trigger_fires_only_at_threshold(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    store = FeedbackExampleStore(engine)
    calls: list[dict] = []

    def fake_retrain(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        return {"status": "deployed"}

    for i in range(4):
        store.add_example(wrong_model_example(f"prompt {i}", "codex"))
        assert not maybe_trigger_retraining(
            engine=engine,
            threshold=5,
            weights_path=tmp_path / "w.json",
            run_async=False,
            retrain=fake_retrain,
        )
    store.add_example(wrong_model_example("prompt 4", "codex"))
    assert maybe_trigger_retraining(
        engine=engine,
        threshold=5,
        weights_path=tmp_path / "w.json",
        run_async=False,
        retrain=fake_retrain,
    )
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Retraining with the golden gate (toy embedder, numpy required)
# ---------------------------------------------------------------------------


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


def test_retrain_deploys_and_marks_processed(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    engine = make_engine(tmp_path)
    store = FeedbackExampleStore(engine)
    for i in range(5):
        store.add_example(wrong_model_example(f"build a login app variant {i}", "codex"))

    weights_path = tmp_path / "weights.json"
    result = retrain_with_feedback(
        engine=engine,
        dataset_path=tmp_path / "unused.jsonl",
        weights_path=weights_path,
        embed=toy_embed,
        embedding_model="toy",
    )

    assert result["status"] == "deployed", result
    assert weights_path.exists()
    meta = json.loads(weights_path.read_text())["metadata"]
    assert meta["feedback_examples"] == 5
    assert store.unprocessed_wrong_model_count() == 0


def test_retrain_rejected_when_golden_accuracy_regresses(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    engine = make_engine(tmp_path)
    store = FeedbackExampleStore(engine)
    store.add_example(wrong_model_example("anything at all", "codex"))

    # Existing live weights claim perfect golden accuracy: any realistic
    # retrain result below that must be rejected.
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps({"metadata": {"golden_accuracy": 1.01}}), encoding="utf-8"
    )

    result = retrain_with_feedback(
        engine=engine,
        dataset_path=tmp_path / "unused.jsonl",
        weights_path=weights_path,
        embed=toy_embed,
        embedding_model="toy",
    )

    assert result["status"] == "rejected"
    assert Path(str(result["rejected_weights"])).exists()
    # Live weights untouched.
    assert json.loads(weights_path.read_text())["metadata"]["golden_accuracy"] == 1.01
    # Examples are marked processed (gate_failed) so the trigger cannot loop.
    assert store.unprocessed_wrong_model_count() == 0


def test_corrected_backend_label_mapping() -> None:
    assert ROUTE_TYPE_BY_BACKEND == {
        "ollama": "local",
        "codex": "coding",
        "claude-code": "reasoning",
    }


# ---------------------------------------------------------------------------
# End to end through the UI API
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    def __init__(self, name: str) -> None:
        self.name = name
        self.cost_type = BackendCostType.LOCAL

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content="answer",
            latency_ms=3,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = BackendRegistry(
        {
            "ollama": FakeAdapter("ollama"),
            "codex": FakeAdapter("codex"),
            "claude-code": FakeAdapter("claude-code"),
        }
    )
    monkeypatch.setattr(
        BackendRegistry,
        "default",
        classmethod(lambda cls, container, cwd=None: registry),
    )


def test_wrong_model_feedback_stores_full_snapshot(
    client: TestClient, fake_backends: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = client.app.state.container
    container.personal_config.preferences.store_feedback_examples = True
    container.personal_config.preferences.feedback_retrain_threshold = 999

    chat = client.post("/api/chat", json={"message": "Say OK only.", "backend": "ollama"})
    assert chat.status_code == 200

    # Find the request id via history.
    session_id = chat.json()["session_id"]
    history = client.get("/api/chat/history", params={"session_id": session_id}).json()
    request_id = history["messages"][-1]["request_id"]

    response = client.post(
        "/api/chat/feedback",
        json={
            "request_id": request_id,
            "rating": "wrong-route",
            "detail": "wrong_model",
            "corrected_backend": "codex",
        },
    )
    assert response.status_code == 200

    store = FeedbackExampleStore(container.memory_repository.engine)
    examples = store.unprocessed_wrong_model_examples()
    assert len(examples) == 1
    example = examples[0]
    assert example.corrected_backend == "codex"
    assert example.prompt == "Say OK only."
    assert example.response_text == "answer"
    assert "<current_user_request>" in example.context_text  # full context captured


def test_good_feedback_stores_nothing(
    client: TestClient, fake_backends: None
) -> None:
    container = client.app.state.container
    container.personal_config.preferences.store_feedback_examples = True

    chat = client.post("/api/chat", json={"message": "Say OK only.", "backend": "ollama"})
    session_id = chat.json()["session_id"]
    history = client.get("/api/chat/history", params={"session_id": session_id}).json()
    request_id = history["messages"][-1]["request_id"]

    client.post("/api/chat/feedback", json={"request_id": request_id, "rating": "good"})

    store = FeedbackExampleStore(container.memory_repository.engine)
    assert store.counts()["total"] == 0


def test_feedback_for_unknown_request_stores_no_example(
    client: TestClient, fake_backends: None
) -> None:
    """Feedback on a request_id Switchboard never saw must not pad the
    retrain threshold with an empty-prompt example."""
    container = client.app.state.container
    container.personal_config.preferences.store_feedback_examples = True

    response = client.post(
        "/api/chat/feedback",
        json={
            "request_id": "req_never_existed",
            "rating": "wrong-route",
            "detail": "wrong_model",
            "corrected_backend": "codex",
        },
    )
    assert response.status_code == 200  # the click itself never fails

    store = FeedbackExampleStore(container.memory_repository.engine)
    assert store.counts()["total"] == 0
    assert store.unprocessed_wrong_model_count() == 0


def test_duplicate_feedback_counts_once_toward_threshold(
    client: TestClient, fake_backends: None
) -> None:
    container = client.app.state.container
    container.personal_config.preferences.store_feedback_examples = True
    container.personal_config.preferences.feedback_retrain_threshold = 999

    chat = client.post("/api/chat", json={"message": "Say OK only.", "backend": "ollama"})
    session_id = chat.json()["session_id"]
    history = client.get("/api/chat/history", params={"session_id": session_id}).json()
    request_id = history["messages"][-1]["request_id"]

    for corrected in ("codex", "claude-code"):
        response = client.post(
            "/api/chat/feedback",
            json={
                "request_id": request_id,
                "rating": "wrong-route",
                "detail": "wrong_model",
                "corrected_backend": corrected,
            },
        )
        assert response.status_code == 200

    store = FeedbackExampleStore(container.memory_repository.engine)
    assert store.counts()["total"] == 1
    assert store.unprocessed_wrong_model_count() == 1
    assert store.unprocessed_wrong_model_examples()[0].corrected_backend == "claude-code"


def test_sensitive_request_leaves_no_context_snapshot(
    client: TestClient, fake_backends: None
) -> None:
    """Privacy: a sensitivity-flagged ask must never persist its assembled
    context, even with store_feedback_examples enabled. A deliberate
    wrong-model thumbs-down still stores the (prompt, label) pair the router
    needs — with empty context."""
    from sqlmodel import Session, select

    from switchboard.app.models.telemetry import RecentContextRecord

    container = client.app.state.container
    container.personal_config.preferences.store_feedback_examples = True
    container.personal_config.preferences.feedback_retrain_threshold = 999

    prompt = "Summarize my diagnosis and medication plan in plain words."
    chat = client.post("/api/chat", json={"message": prompt, "backend": "auto"})
    assert chat.status_code == 200

    engine = container.memory_repository.engine
    session_id = chat.json()["session_id"]
    history = client.get("/api/chat/history", params={"session_id": session_id}).json()
    request_id = history["messages"][-1]["request_id"]

    # The reroute fired and no context snapshot was written.
    metric = container.backend_metrics_repository.get(request_id)
    assert metric is not None and metric.metadata.get("private_mode_rerouted") is True
    with Session(engine) as db:
        assert db.exec(select(RecentContextRecord)).all() == []

    response = client.post(
        "/api/chat/feedback",
        json={
            "request_id": request_id,
            "rating": "wrong-route",
            "detail": "wrong_model",
            "corrected_backend": "codex",
        },
    )
    assert response.status_code == 200

    examples = FeedbackExampleStore(engine).unprocessed_wrong_model_examples()
    assert len(examples) == 1
    assert examples[0].prompt == prompt  # deliberate user choice: router pair kept
    assert examples[0].context_text == ""  # but never the assembled context
