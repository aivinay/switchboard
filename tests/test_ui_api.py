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
from switchboard.app.models.telemetry import BackendMetricRecord

ROOT = Path(__file__).resolve().parents[1]


class RecordingAdapter(AgentAdapter):
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        fail_with_exception: bool = False,
        cost_type: BackendCostType = BackendCostType.LOCAL,
    ) -> None:
        self.name = name
        self.available = available
        self.fail_with_exception = fail_with_exception
        self.cost_type = cost_type
        self.calls: list[SwitchboardRequest] = []

    def is_available(self) -> bool:
        return self.available

    def availability(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            available=self.available,
            cost_type=self.cost_type,
            path=f"/fake/{self.name}" if self.available else None,
            warning=None if self.available else f"{self.name} is unavailable",
        )

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.calls.append(request)
        if self.fail_with_exception:
            raise RuntimeError("simulated adapter failure with internal details")
        if not self.available:
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                success=False,
                error_message=f"{self.name} CLI is unavailable: {self.name} not found",
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} answered",
            selected_model=f"{self.name}/test",
            latency_ms=7,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


@pytest.fixture
def fake_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, RecordingAdapter]:
    monkeypatch.setenv("SWITCHBOARD_HTTP_ENABLE_CLI_BACKENDS", "true")
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    registry = BackendRegistry(adapters)
    monkeypatch.setattr(
        BackendRegistry,
        "default",
        classmethod(lambda cls, container, cwd=None: registry),
    )
    return adapters


def test_ui_chat_blocks_subscription_cli_backends_by_default(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SWITCHBOARD_HTTP_ENABLE_CLI_BACKENDS", raising=False)

    response = client.post("/api/chat", json={"message": "Say OK only.", "backend": "codex"})

    assert response.status_code == 403
    assert "disabled on the HTTP API" in response.json()["detail"]["message"]


@pytest.mark.parametrize(
    ("backend", "prompt", "expected_backend", "expected_display_model", "expected_adapter"),
    [
        ("auto", "Debug this repo and suggest a fix.", "codex", "Codex", "codex"),
        ("codex", "Say OK only.", "codex", "Codex", "codex"),
        ("claude", "Explain the architecture.", "claude-code", "Claude", "claude-code"),
        ("ollama", "Say OK only.", "ollama", "Ollama", "ollama"),
    ],
)
def test_ui_chat_api_handles_supported_backends(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
    backend: str,
    prompt: str,
    expected_backend: str,
    expected_display_model: str,
    expected_adapter: str,
) -> None:
    response = client.post("/api/chat", json={"message": prompt, "backend": backend})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == f"{expected_adapter} answered"
    assert body["backend"] == expected_backend
    assert body["display_model"] == expected_display_model
    assert body["session_id"].startswith("session_")
    assert len(fake_adapters[expected_adapter].calls) == 1


def test_ui_auto_uses_router_and_forced_backend_bypasses_auto(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    auto_response = client.post(
        "/api/chat",
        json={"message": "Debug this repo and suggest a fix.", "backend": "auto"},
    )
    forced_response = client.post(
        "/api/chat",
        json={"message": "Debug this repo and suggest a fix.", "backend": "ollama"},
    )

    assert auto_response.status_code == 200
    assert auto_response.json()["backend"] == "codex"
    assert forced_response.status_code == 200
    assert forced_response.json()["backend"] == "ollama"
    assert len(fake_adapters["codex"].calls) == 1
    assert len(fake_adapters["ollama"].calls) == 1


def test_ui_chat_api_rejects_empty_prompt(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "   ", "backend": "auto"})

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Enter a message before sending."


def test_ui_chat_api_rejects_invalid_backend(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "Hello", "backend": "gpt"})

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Choose Auto, Codex, Claude, or Ollama."


@pytest.mark.parametrize(
    ("ui_backend", "adapter_name", "expected_message"),
    [
        ("codex", "codex", "Codex is not available"),
        ("claude", "claude-code", "Claude is not available"),
        ("ollama", "ollama", "Ollama is not running"),
    ],
)
def test_ui_chat_api_returns_clean_errors_for_unavailable_backends(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
    ui_backend: str,
    adapter_name: str,
    expected_message: str,
) -> None:
    fake_adapters[adapter_name].available = False

    response = client.post(
        "/api/chat",
        json={"message": "Say OK only.", "backend": ui_backend},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert expected_message in detail["message"]
    assert detail["display_model"] in {"Codex", "Claude", "Ollama"}
    assert "not found" not in detail["message"]
    assert "stderr" not in detail["message"].lower()


def test_ui_chat_api_hides_adapter_exception_details(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    fake_adapters["codex"].fail_with_exception = True

    response = client.post(
        "/api/chat",
        json={"message": "Debug this repo.", "backend": "codex"},
    )

    assert response.status_code == 502
    message = response.json()["detail"]["message"]
    assert message == "Something went wrong. Please try again or choose another model."
    assert "RuntimeError" not in message
    assert "simulated" not in message


def test_ui_chat_api_records_backend_metrics(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Debug this repo and suggest a fix.", "backend": "auto"},
    )

    assert response.status_code == 200
    records = client.app.state.container.backend_metrics_repository.list()
    assert len(records) == 1
    assert records[0].backend == "codex"
    assert records[0].success is True
    assert records[0].metadata["surface"] == "ui"
    assert records[0].metadata["requested_backend"] == "auto"


def test_ui_quota_endpoint_reports_declared_budget_usage(client: TestClient) -> None:
    container = client.app.state.container
    container.personal_config.quota.codex_calls_per_5h = 2
    container.backend_metrics_repository.add(
        BackendMetricRecord(
            request_id="req_ui_quota",
            backend="codex",
            selected_model="codex/test",
            project="ui",
            prompt_char_count=10,
            latency_ms=12,
            success=True,
            routing_reason="test quota metric",
            cost_type="subscription",
            estimated_cost_usd=0.0,
            private_mode=True,
        )
    )

    response = client.get("/api/quota")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["windows"]["codex"]["used"] == 1
    assert body["windows"]["codex"]["budget"] == 2
    assert body["windows"]["claude-code"]["budget"] is None


def test_ui_version_endpoint_reads_cached_update_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "config-home" / "switchboard"
    cache_dir.mkdir(parents=True)
    (cache_dir / "update-check.json").write_text(
        json.dumps({"latest": "9.9.9", "checked_at": "2026-07-02T00:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SWITCHBOARD_CONFIG_HOME", str(tmp_path / "config-home"))

    response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {
        "installed": "0.3.0",
        "latest": "9.9.9",
        "update_available": True,
    }


def test_ui_backends_status_endpoint_lists_auto_first(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.get("/api/backends/status")

    assert response.status_code == 200
    body = response.json()
    values = [option["value"] for option in body["options"]]
    assert values == ["auto", "codex", "claude", "ollama"]
    assert body["options"][0]["label"] == "Auto"
    assert body["options"][0]["available"] is True
    assert body["options"][1]["available"] is True
    assert body["private_mode"] is True


def test_ui_dashboard_endpoint_uses_recorded_metrics(client: TestClient) -> None:
    container = client.app.state.container
    container.backend_metrics_repository.add(
        BackendMetricRecord(
            request_id="req_dashboard_local",
            backend="ollama",
            selected_model="ollama/test",
            project="ui",
            prompt_char_count=40,
            latency_ms=12,
            success=True,
            routing_reason="local",
            cost_type="local",
            estimated_cost_usd=0.0,
            private_mode=True,
            metadata_json=json.dumps(
                {
                    "compression_tokens_saved": 4,
                    "context_compression_tokens_saved": 30,
                }
            ),
        )
    )
    container.backend_metrics_repository.add(
        BackendMetricRecord(
            request_id="req_dashboard_premium",
            backend="codex",
            selected_model="codex/test",
            project="ui",
            prompt_char_count=80,
            latency_ms=20,
            success=True,
            routing_reason="coding",
            cost_type="subscription",
            estimated_cost_usd=0.0,
            private_mode=True,
        )
    )

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    body = response.json()
    assert body["premium_calls"] == 1
    assert body["premium_calls_avoided_vs_always_premium"] == 1
    assert body["estimated_tokens_saved"]["compression"] == 34
    assert body["estimated_tokens_saved"]["routing"] == 10
    assert body["usage_by_backend"] == {"codex": 1, "ollama": 1}
    assert len(body["last_7_days"]) == 7


def test_ui_time_question_uses_tool_grounding_with_selected_model(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Time in India", "backend": "auto"},
    )

    assert response.status_code == 200
    body = response.json()
    # Tool-grounded answers route to the free local model for formatting.
    assert body["backend"] == "ollama"
    assert body["display_model"] == "Ollama"
    assert body["session_id"].startswith("session_")
    assert "ollama answered" in body["answer"]
    assert len(fake_adapters["ollama"].calls) == 1
    assert "The current time in India" in fake_adapters["ollama"].calls[0].prompt


def test_ui_weather_question_passes_through_without_provider_warning(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Weather in Dubai", "backend": "auto"},
    )

    assert response.status_code == 200
    body = response.json()
    # Live-data without a provider routes to the free local model and carries
    # an anti-fabrication instruction (dogfood regression).
    assert body["backend"] == "ollama"
    assert body["display_model"] == "Ollama"
    assert body["session_id"].startswith("session_")
    assert "ollama answered" in body["answer"]
    assert len(fake_adapters["ollama"].calls) == 1
    sent_prompt = fake_adapters["ollama"].calls[0].prompt
    assert "Weather in Dubai" in sent_prompt
    assert "Do not invent specific" in sent_prompt


def test_ui_stock_question_passes_through_without_provider_warning(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "ServiceNow stock price", "backend": "auto"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "ollama"
    assert body["display_model"] == "Ollama"
    assert "ollama answered" in body["answer"]
    sent_prompt = fake_adapters["ollama"].calls[0].prompt
    assert "ServiceNow stock price" in sent_prompt
    assert "Do not invent specific" in sent_prompt


def test_ui_chat_api_reuses_provided_session_id(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    first = client.post(
        "/api/chat",
        json={"message": "Remember this codebase detail.", "backend": "codex"},
    )
    session_id = first.json()["session_id"]
    second = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Use that detail in a design review.",
            "backend": "claude",
        },
    )

    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    assert "Remember this codebase detail." in fake_adapters["claude-code"].calls[0].prompt


def test_ui_chat_api_response_hides_internal_metadata(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat",
        json={"message": "Review this architecture.", "backend": "claude"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"session_id", "answer", "backend", "display_model"}
    assert body["backend"] == "claude-code"
    assert body["display_model"] == "Claude"
    serialized = json.dumps(body)
    assert "routing_reason" not in serialized
    assert "capabilities" not in serialized
    assert "metrics" not in serialized
    assert "stdout" not in serialized
    assert "stderr" not in serialized
    assert "You are answering inside Switchboard" not in serialized


def parse_stream_events(body: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_ui_chat_stream_returns_metadata_chunks_and_done(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat/stream",
        json={"message": "Debug this repo and suggest a fix.", "backend": "auto"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = parse_stream_events(response.text)
    assert [event["type"] for event in events] == ["start", "metadata", "chunk", "done"]
    assert str(events[0]["session_id"]).startswith("session_")
    assert events[1]["session_id"] == events[0]["session_id"]
    assert events[1]["backend"] == "codex"
    assert events[1]["display_model"] == "Codex"
    assert events[2]["text"] == "codex answered"
    assert events[3]["display_model"] == "Codex"
    assert len(fake_adapters["codex"].calls) == 1


def test_ui_stream_does_not_render_hidden_runtime_context(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    response = client.post(
        "/api/chat/stream",
        json={"message": "Debug this repo and suggest a fix.", "backend": "auto"},
    )

    assert response.status_code == 200
    assert "[Switchboard runtime context]" not in response.text
    assert "Current local datetime:" not in response.text
    assert "You are replying to the user through Switchboard" not in response.text
    assert "You are replying to the user through Switchboard" in fake_adapters[
        "codex"
    ].calls[0].prompt


def test_ui_chat_stream_returns_clean_error_event(
    client: TestClient,
    fake_adapters: dict[str, RecordingAdapter],
) -> None:
    fake_adapters["claude-code"].available = False

    response = client.post(
        "/api/chat/stream",
        json={"message": "Review the design.", "backend": "claude"},
    )

    assert response.status_code == 200
    events = parse_stream_events(response.text)
    assert [event["type"] for event in events] == ["start", "error"]
    assert events[1]["display_model"] == "Claude"
    assert events[1]["message"] == (
        "Claude is not available. Please install and authenticate Claude Code, "
        "or choose another model."
    )
    assert "not found" not in str(events[1]["message"])


def test_ui_chat_stream_rejects_empty_and_invalid_requests(client: TestClient) -> None:
    empty = client.post("/api/chat/stream", json={"message": "   ", "backend": "auto"})
    invalid = client.post("/api/chat/stream", json={"message": "Hello", "backend": "gpt"})

    assert empty.status_code == 400
    assert empty.json()["detail"]["message"] == "Enter a message before sending."
    assert invalid.status_code == 400
    assert invalid.json()["detail"]["message"] == "Choose Auto, Codex, Claude, or Ollama."


def test_ui_static_files_exist_and_call_chat_api(client: TestClient) -> None:
    static_dir = ROOT / "switchboard" / "app" / "static"
    index = static_dir / "index.html"
    state_js = static_dir / "state.js"
    overlays_js = static_dir / "overlays.js"
    app_js = static_dir / "app.js"

    assert index.exists()
    assert state_js.exists()
    assert overlays_js.exists()
    assert app_js.exists()

    html = index.read_text(encoding="utf-8")
    assert "<title>Switchboard</title>" in html
    assert "<h1>Switchboard</h1>" in html
    assert "Personal AI Switchboard" not in html
    assert "personal_ai_switchboard" not in html
    assert 'id="model-menu"' in html
    assert 'id="dashboard-toggle"' in html
    assert 'id="quota-meters"' in html
    assert 'id="private-mode"' in html
    assert "Backend" not in html
    assert "Ask Switchboard..." in html
    assert html.index('id="model-picker-button"') < html.index("<h1>Switchboard</h1>")
    assert 'aria-label="Send message"' in html
    assert "<button id=\"send\"" in html
    assert ">Send<" not in html
    assert html.index("/ui/static/state.js") < html.index("/ui/static/overlays.js")
    assert html.index("/ui/static/overlays.js") < html.index("/ui/static/app.js")

    state = state_js.read_text(encoding="utf-8")
    assert "window.SB" in state
    assert "switchboard.session_id" in state
    assert "openOverlayStack" in state
    assert "switchboard.feedback.enable_nudge_seen" in state

    overlays = overlays_js.read_text(encoding="utf-8")
    assert "SB.dismissableStack" in overlays
    assert 'event.key === "Escape"' in overlays
    assert "closeTop" in overlays

    javascript = app_js.read_text(encoding="utf-8")
    assert "window.SB" in javascript
    assert "SB.dismissableStack.register" in javascript
    assert "/api/chat/stream" in javascript
    assert "Thinking..." in javascript
    assert "display_model" in javascript
    assert "streamAssistantResponse" in javascript
    assert "handleStreamEvent" in javascript
    assert "addAssistantMessage" in javascript
    assert "renderMarkdown" in javascript
    assert "makeMetaRow" in javascript
    assert "loadHistory" in javascript
    assert "/api/chat/history" in javascript
    assert "/api/chat/feedback" in javascript
    assert "/api/backends/status" in javascript
    assert "/api/dashboard" in javascript
    assert "/api/quota" in javascript
    assert "rememberSession" in javascript
    assert "session_id: sessionId" in javascript
    assert "updateSendState" in javascript
    assert "input.value.trim().length === 0" in javascript
    assert "renderModelOptions" in javascript
    assert "appendRoutingChips" in javascript
    # Internal backend ids appear only as API payload values (feedback
    # correction buttons); visible labels remain friendly names.
    assert '["Claude", "claude-code"]' in javascript
    assert "feedback-popover" in javascript
    assert 'method: "DELETE"' in javascript
    assert 'rating: "bad"' in javascript
    assert "corrected_backend" in javascript
    assert "nudge_enable_examples" in javascript
    page = client.get("/ui")
    assert page.status_code == 200
    assert "Switchboard" in page.text


def test_ui_dropdown_content_and_selection_logic() -> None:
    static_dir = ROOT / "switchboard" / "app" / "static"
    javascript = (static_dir / "app.js").read_text(encoding="utf-8")

    assert "Routes automatically" in javascript
    assert "Best for coding tasks" in javascript
    assert "Good for reasoning and design" in javascript
    assert "Runs locally" in javascript
    assert 'button.classList.add("selected")' in javascript
    assert 'button.setAttribute("aria-selected"' in javascript
    assert "chooseModel(option.value)" in javascript
    assert 'selectedModel.textContent = modelLabels[value] || value' in javascript
    assert 'setMenuOpen(false)' in javascript


def test_ui_layout_keeps_composer_stable() -> None:
    css = (
        ROOT / "switchboard" / "app" / "static" / "styles.css"
    ).read_text(encoding="utf-8")

    assert "height: 100dvh;" in css
    assert "grid-template-rows: auto minmax(0, 1fr) auto;" in css
    assert ".messages {" in css
    assert "overflow-y: auto;" in css
    assert "scrollbar-width: none;" in css
    assert ".messages::-webkit-scrollbar" in css
    assert "min-height: 0;" in css
    assert ".composer {" in css
    assert "grid-template-rows: auto auto;" in css
    assert ".composer-footer {" in css
    assert "border-radius: 999px;" in css
    assert "grid-template-columns: 1fr auto 1fr;" in css
    assert "top: calc(100% + 8px);" in css
    assert "margin-top: 8px;" in css
