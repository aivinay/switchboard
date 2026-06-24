"""Tests for UI v2 features: history, routing transparency, and feedback."""

from __future__ import annotations

import json

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


class FakeAdapter(AgentAdapter):
    def __init__(self, name: str, *, cost_type: BackendCostType = BackendCostType.LOCAL) -> None:
        self.name = name
        self.cost_type = cost_type

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} says hello",
            selected_model=f"{self.name}/test-model",
            latency_ms=42,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = BackendRegistry(
        {
            "ollama": FakeAdapter("ollama"),
            "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
            "claude-code": FakeAdapter(
                "claude-code", cost_type=BackendCostType.SUBSCRIPTION
            ),
        }
    )
    monkeypatch.setattr(
        BackendRegistry,
        "default",
        classmethod(lambda cls, container, cwd=None: registry),
    )


def stream_events(client: TestClient, message: str, session_id: str | None = None) -> list[dict]:
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": message, "backend": "ollama", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_raw()).decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_stream_metadata_includes_routing_transparency(
    client: TestClient, fake_backends: None
) -> None:
    events = stream_events(client, "Say OK only.")
    metadata = next(event for event in events if event["type"] == "metadata")
    done = next(event for event in events if event["type"] == "done")

    for event in (metadata, done):
        assert event["request_id"].startswith("req")
        assert event["latency_ms"] == 42
        assert event["cost_type"] == "local"
        assert event["selected_model"] == "ollama/test-model"
        assert event["routing_reason"]


def test_history_returns_past_turns_with_request_ids(
    client: TestClient, fake_backends: None
) -> None:
    events = stream_events(client, "Remember the number 7.")
    session_id = events[0]["session_id"]
    stream_events(client, "And the number 8.", session_id=session_id)

    response = client.get("/api/chat/history", params={"session_id": session_id})
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assistant = payload["messages"][1]
    assert assistant["content"] == "ollama says hello"
    assert assistant["request_id"]
    assert assistant["display_model"] == "Ollama"


def test_history_for_unknown_session_is_empty(client: TestClient) -> None:
    response = client.get("/api/chat/history", params={"session_id": "session_nope"})
    assert response.status_code == 200
    assert response.json() == {"session_id": "session_nope", "messages": []}


def test_feedback_endpoint_records_rating(client: TestClient, fake_backends: None) -> None:
    events = stream_events(client, "Say OK only.")
    request_id = next(event for event in events if event["type"] == "done")["request_id"]

    response = client.post(
        "/api/chat/feedback",
        json={"request_id": request_id, "rating": "good"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == request_id
    assert payload["rating"] == "good"


def test_feedback_rejects_unknown_rating(client: TestClient) -> None:
    response = client.post(
        "/api/chat/feedback",
        json={"request_id": "req_x", "rating": "amazing"},
    )
    assert response.status_code == 400


def test_feedback_rejects_invalid_corrected_backend_before_storing(
    client: TestClient, fake_backends: None
) -> None:
    """A wrong-model verdict with an unusable correction is a 400 that names
    the valid values, and nothing — not even the rating — is stored."""
    from sqlmodel import Session, select

    from switchboard.app.models.telemetry import (
        FeedbackExampleRecord,
        FeedbackRecord,
    )

    container = client.app.state.container
    container.personal_config.preferences.store_feedback_examples = True

    events = stream_events(client, "Say OK only.")
    request_id = next(event for event in events if event["type"] == "done")["request_id"]

    for corrected in ("gpt-5", None):
        response = client.post(
            "/api/chat/feedback",
            json={
                "request_id": request_id,
                "rating": "wrong-route",
                "detail": "wrong_model",
                "corrected_backend": corrected,
            },
        )
        assert response.status_code == 400
        message = response.json()["detail"]["message"]
        assert "ollama" in message and "codex" in message and "claude-code" in message

    with Session(container.memory_repository.engine) as session:
        assert session.exec(select(FeedbackRecord)).all() == []
        assert session.exec(select(FeedbackExampleRecord)).all() == []
