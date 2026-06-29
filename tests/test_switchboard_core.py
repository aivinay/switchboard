from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import httpx
import pytest

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.cli_agents import ClaudeCodeCliAdapter, CodexCliAdapter
from switchboard.app.backends.ollama_backend import OllamaAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    BackendRouteDecision,
    SwitchboardRequest,
    SwitchboardResponse,
    backend_display_name,
)
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.models.telemetry import BackendMetricRecord
from switchboard.app.providers.base import ProviderResponse
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.services.container import build_container
from switchboard.app.services.cost import CostEstimator
from switchboard.app.services.semantic_memory import EmbeddingUnavailableError
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.app.storage.repositories import BackendMetricsRepository
from switchboard.cli import (
    ask_command,
    backend_error_hint,
    backends_command,
    doctor_command,
    make_parser,
    metrics_command,
    route_command,
    train_dispatcher_command,
    train_router_command,
    train_sensitivity_command,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeAdapter(AgentAdapter):
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        cost_type: BackendCostType = BackendCostType.LOCAL,
    ) -> None:
        self.name = name
        self.available = available
        self.cost_type = cost_type

    def is_available(self) -> bool:
        return self.available

    def availability(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            available=self.available,
            cost_type=self.cost_type,
            path=f"/fake/{self.name}" if self.available else None,
            warning=None if self.available else f"{self.name} unavailable",
        )

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        if not self.available:
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                latency_ms=1,
                success=False,
                error_message=f"{self.name} unavailable",
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} answered",
            stdout=f"{self.name} answered",
            latency_ms=12,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


class RaisingAdapter(FakeAdapter):
    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        raise RuntimeError("boom")


class PromptEchoErrorAdapter(FakeAdapter):
    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            latency_ms=3,
            success=False,
            error_message=(
                "Runtime context:\n"
                "- Current UTC time: test\n"
                "Current request:\n"
                f"{request.prompt}"
            ),
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


class FakeRuntime:
    enabled = True

    def list_installed_models(self) -> set[str]:
        return {"ollama/llama3.2:3b"}


def switchboard_request(prompt: str = "Explain this") -> SwitchboardRequest:
    return SwitchboardRequest(request_id="req_test", prompt=prompt)


def make_core_service(
    tmp_path: Path,
    registry: BackendRegistry,
) -> SwitchboardCoreService:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'core.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    return SwitchboardCoreService(
        registry=registry,
        metrics=container.backend_metrics_repository,
        container=container,
    )


def test_codex_availability_detection(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/codex")
    assert CodexCliAdapter().is_available()

    monkeypatch.setattr("shutil.which", lambda executable: None)
    assert not CodexCliAdapter().is_available()


def test_claude_code_availability_detection(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/claude")
    assert ClaudeCodeCliAdapter().is_available()

    monkeypatch.setattr("shutil.which", lambda executable: None)
    assert not ClaudeCodeCliAdapter().is_available()


def test_codex_command_construction_uses_safe_subprocess_args(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/codex")
    command = CodexCliAdapter(cwd=tmp_path).build_command(switchboard_request("Debug tests"))

    assert command[:5] == [
        "/usr/bin/codex",
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
    ]
    assert "--ask-for-approval" not in command
    assert "--cd" in command
    assert command[-1] == "Debug tests"


def test_codex_reports_configured_default_model(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-5.5"\n', encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/codex")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="OK\n",
            stderr="",
        ),
    )

    response = CodexCliAdapter(config_path=config_path).ask(switchboard_request())

    assert response.success
    assert response.selected_model == "gpt-5.5"


def test_claude_command_construction_uses_non_interactive_print(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/claude")
    command = ClaudeCodeCliAdapter(cwd=tmp_path).build_command(
        switchboard_request("Explain architecture")
    )

    assert command[0] == "/usr/bin/claude"
    assert "--print" in command
    assert "--output-format=json" in command
    assert "--no-session-persistence" in command
    assert "--disallowedTools=Edit,Write,Bash" in command
    assert "--add-dir" not in command
    assert command[-1] == "Explain architecture"


def test_cli_adapter_captures_stdout_and_stderr(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/codex")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="hello\n",
            stderr="note\n",
        ),
    )

    response = CodexCliAdapter(config_path=tmp_path / "missing.toml").ask(switchboard_request())

    assert response.success
    assert response.content == "hello"
    assert response.selected_model == "codex/default"
    assert response.stdout == "hello\n"
    assert response.stderr == "note\n"
    assert response.exit_code == 0


def test_cli_adapter_reports_requested_model(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="hello\n",
            stderr="",
        ),
    )

    response = ClaudeCodeCliAdapter().ask(
        SwitchboardRequest(request_id="req_test", prompt="hello", model="sonnet")
    )

    assert response.success
    assert response.selected_model == "sonnet"


def test_claude_adapter_parses_json_result_and_model_usage(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/claude")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=(
                '{"result":"OK","modelUsage":{"claude-opus-4-6[1m]":'
                '{"inputTokens":3,"outputTokens":4}}}'
            ),
            stderr="",
        ),
    )

    response = ClaudeCodeCliAdapter().ask(switchboard_request())

    assert response.success
    assert response.content == "OK"
    assert response.selected_model == "claude-opus-4-6"


def test_cli_adapter_reports_non_zero_exit(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/codex")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=2,
            stdout="",
            stderr="failed",
        ),
    )

    response = CodexCliAdapter().ask(switchboard_request())

    assert not response.success
    assert response.stderr == "failed"
    assert response.exit_code == 2
    assert response.error_message == "failed"


def test_cli_adapter_reports_timeout(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/codex")

    def timeout(*args, **kwargs):  # noqa: ANN002, ANN003
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr(subprocess, "run", timeout)

    response = CodexCliAdapter().ask(
        SwitchboardRequest(request_id="req_test", prompt="x", timeout_s=1)
    )

    assert not response.success
    assert response.exit_code is None
    assert response.error_message == "codex timed out after 1s."


def test_ollama_backend_adapter_uses_existing_provider(monkeypatch) -> None:
    async def fake_complete_chat(
        self,
        request: NormalizedRequest,
        model_profile,
    ) -> ProviderResponse:  # noqa: ANN001
        return ProviderResponse(
            content="local answer",
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=2,
        )

    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fake_complete_chat)
    adapter = OllamaAdapter(
        catalogue=ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml"),
        provider=OllamaProviderAdapter(),
        runtime=FakeRuntime(),  # type: ignore[arg-type]
        cost_estimator=CostEstimator(),
    )

    response = adapter.ask(switchboard_request())

    assert response.success
    assert response.backend == "ollama"
    assert response.content == "local answer"
    # Default local chat model: llama3.2:3b (first enabled local chat model
    # in the catalogue; fast beats heavy for the default experience).
    assert response.selected_model == "ollama/llama3.2:3b"


def test_backend_metrics_records_success_failure_and_summary(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'metrics.db'}")
    init_db(engine)
    repo = BackendMetricsRepository(engine)

    repo.add(
        BackendMetricRecord(
            request_id="req_success",
            backend="ollama",
            selected_model="ollama/llama3.2:3b",
            project="personal",
            prompt_char_count=12,
            latency_ms=10,
            success=True,
            routing_reason="Default baseline prefers Ollama.",
            cost_type="local",
            estimated_cost_usd=0.0,
            private_mode=True,
        )
    )
    repo.add(
        BackendMetricRecord(
            request_id="req_failed",
            backend="codex",
            project="personal",
            prompt_char_count=12,
            latency_ms=20,
            success=False,
            error_message="codex unavailable",
            exit_code=1,
            routing_reason="Coding task prefers Codex.",
            cost_type="subscription",
            estimated_cost_usd=0.0,
            private_mode=True,
        )
    )

    records = repo.list(limit=10)
    summary = repo.summary()

    assert len(records) == 2
    assert summary["total_requests"] == 2
    assert summary["requests_by_backend"] == {"ollama": 1, "codex": 1}
    assert summary["success_rate_by_backend"] == {"ollama": 1.0, "codex": 0.0}
    assert summary["average_latency_ms_by_backend"] == {"ollama": 10.0, "codex": 20.0}
    assert summary["recent_errors"][0]["request_id"] == "req_failed"  # type: ignore[index]


def test_backend_metrics_redacts_prompt_like_provider_errors(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'metrics_redaction.db'}")
    init_db(engine)
    repo = BackendMetricsRepository(engine)
    repo.add(
        BackendMetricRecord(
            request_id="req_redacted",
            backend="codex",
            selected_model="codex/default",
            project="personal",
            prompt_char_count=18,
            latency_ms=10,
            success=False,
            error_message=(
                "Runtime context:\n"
                "Recent conversation:\n"
                "Current request:\n"
                "private prompt body"
            ),
            routing_reason="User selected backend codex.",
            cost_type="subscription",
            private_mode=True,
        )
    )

    record = repo.list(limit=1)[0]
    summary = repo.summary()

    assert "private prompt body" not in (record.error_message or "")
    assert "redacted" in (record.error_message or "")
    recent_error = summary["recent_errors"][0]  # type: ignore[index]
    assert "private prompt body" not in recent_error["error_message"]  # type: ignore[index]


def test_backend_router_forced_selection(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex")}),
    )

    decision = service.route(switchboard_request(), forced_backend="codex")

    assert decision.backend == "codex"
    assert "User selected backend codex" in decision.routing_reason


def test_backend_router_prefers_codex_for_coding(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex")}),
    )

    decision = service.route(switchboard_request("Debug this repo test failure"))

    assert decision.backend == "codex"
    assert "prefers Codex" in decision.routing_reason


@pytest.mark.parametrize(
    "prompt",
    [
        "fix this failing test",
        "fix this failing Python test",
        "implement this UI change",
        "debug this traceback",
        "analyze this repo",
        "create a prompt for Codex",
        "create a prompt for Codex to update the code",
        "run/debug/update code",
    ],
)
def test_phase_a_auto_routes_coding_tasks_to_codex(tmp_path: Path, prompt: str) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex"),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    decision = service.route(switchboard_request(prompt))

    assert decision.backend == "codex"
    assert decision.selected_backend == "codex"
    assert decision.display_model == "Codex"
    assert decision.route_type == "coding"
    assert not decision.fallback_used
    assert decision.fallback_from is None


def test_backend_router_prefers_claude_for_architecture(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"claude-code": FakeAdapter("claude-code")}),
    )

    decision = service.route(switchboard_request("Do a system design review"))

    assert decision.backend == "claude-code"
    assert "prefers Claude Code" in decision.routing_reason


@pytest.mark.parametrize(
    "prompt",
    [
        "review this architecture",
        "think through tradeoffs",
        "create a research plan",
        "does this system design make sense",
        "act as a principal engineer and review this",
        "principal engineer review",
        "distributed systems design",
    ],
)
def test_phase_a_auto_routes_reasoning_tasks_to_claude(tmp_path: Path, prompt: str) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex"),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    decision = service.route(switchboard_request(prompt))

    assert decision.backend == "claude-code"
    assert decision.selected_backend == "claude-code"
    assert decision.display_model == "Claude"
    assert decision.route_type == "reasoning"


@pytest.mark.parametrize(
    "prompt",
    [
        "answer locally",
        "hi",
        "hello",
        "thanks",
        "private summary",
        "use local model",
        "cheap quick answer",
        "summarize this short text",
        "summarize this short text locally",
    ],
)
def test_phase_a_auto_routes_local_private_simple_tasks_to_ollama(
    tmp_path: Path,
    prompt: str,
) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex"),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    decision = service.route(switchboard_request(prompt))

    assert decision.backend == "ollama"
    assert decision.display_model == "Ollama"
    assert decision.route_type == "local"


def test_phase_a_unknown_prompt_defaults_local_first(tmp_path: Path) -> None:
    # Product decision (2026-06-12): unknown tasks fail CLOSED to the free
    # local model; premium backends are a deliberate exception, never the
    # default for unclassifiable prompts.
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex"),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    decision = service.route(switchboard_request("what should I do next?"))

    assert decision.backend == "ollama"
    assert decision.route_type == "unknown"
    assert "local-first default" in decision.routing_reason


def test_backend_router_falls_back_when_preferred_unavailable(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", available=False),
            }
        ),
    )

    decision = service.route(switchboard_request("Refactor this repo"))

    assert decision.backend == "ollama"
    assert decision.fallback_used
    assert "fell back to Ollama" in decision.routing_reason


def test_backend_router_prefers_claude_before_ollama_for_codex_fallback(
    tmp_path: Path,
) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", available=False),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    decision = service.route(switchboard_request("Refactor this repo"))

    assert decision.backend == "claude-code"
    assert decision.fallback_used
    assert "fell back to Claude" in decision.routing_reason


def test_phase_a_coding_falls_back_from_codex_to_claude_then_ollama(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", available=False),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )
    first_decision = service.route(switchboard_request("why is this test failing"))

    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", available=False),
                "claude-code": FakeAdapter("claude-code", available=False),
            }
        ),
    )
    second_decision = service.route(switchboard_request("why is this test failing"))

    assert first_decision.backend == "claude-code"
    assert first_decision.fallback_used
    assert first_decision.fallback_from == "codex"
    assert second_decision.backend == "ollama"
    assert second_decision.fallback_used
    assert second_decision.fallback_from == "codex"


def test_phase_a_reasoning_and_local_fallback_orders(tmp_path: Path) -> None:
    reasoning_service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex"),
                "claude-code": FakeAdapter("claude-code", available=False),
            }
        ),
    )
    local_service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama", available=False),
                "codex": FakeAdapter("codex"),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    reasoning_decision = reasoning_service.route(switchboard_request("review this architecture"))
    local_decision = local_service.route(switchboard_request("answer locally"))

    assert reasoning_decision.backend == "codex"
    assert reasoning_decision.fallback_from == "claude-code"
    assert local_decision.backend == "claude-code"
    assert local_decision.fallback_from == "ollama"


def test_phase_a_forced_backend_does_not_fallback_when_unavailable(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", available=False),
                "claude-code": FakeAdapter("claude-code"),
            }
        ),
    )

    decision = service.route(
        switchboard_request("review this architecture"),
        forced_backend="codex",
    )
    response = service.ask("review this architecture", backend="codex")

    assert decision.backend == "codex"
    assert decision.route_type == "forced"
    assert decision.forced_backend
    assert not decision.fallback_used
    assert not response.success
    assert "unavailable" in (response.error_message or "")


def test_phase_a_no_backends_available_returns_clean_error(tmp_path: Path) -> None:
    service = make_core_service(tmp_path, BackendRegistry({}))

    response = service.ask("Debug this repo", backend="auto")

    assert not response.success
    assert response.backend == "codex"
    assert response.error_message == (
        "No configured Switchboard model is available. Install Codex, "
        "Claude Code, or Ollama and try again."
    )


@pytest.mark.parametrize("backend", ["codex", "claude-code", "ollama"])
def test_phase_a_forced_backend_uses_only_selected_backend(
    tmp_path: Path,
    backend: str,
) -> None:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex"),
        "claude-code": FakeAdapter("claude-code"),
    }
    service = make_core_service(tmp_path, BackendRegistry(adapters))

    response = service.ask("Debug this repo", backend=backend)

    assert response.backend == backend
    assert response.success
    assert response.routing_reason == f"User selected backend {backend}."


def test_phase_a_display_model_mapping() -> None:
    assert backend_display_name("codex") == "Codex"
    assert backend_display_name("claude-code") == "Claude"
    assert backend_display_name("ollama") == "Ollama"


def test_backend_ask_records_routing_reason(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"codex": FakeAdapter("codex"), "ollama": FakeAdapter("ollama")}),
    )

    response = service.ask("Debug this repo test failure", backend="auto")
    records = service.metrics_list(limit=1)

    assert response.success
    assert records[0].backend == "codex"  # type: ignore[attr-defined]
    assert "prefers Codex" in records[0].routing_reason  # type: ignore[attr-defined,operator]


def test_phase_a_metrics_record_auto_route_metadata(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"codex": FakeAdapter("codex"), "ollama": FakeAdapter("ollama")}),
    )

    response = service.ask("Debug this repo test failure", backend="auto")
    records = service.metrics_list(limit=1)
    metadata = records[0].metadata

    assert response.success
    assert metadata["selected_backend"] == "codex"
    assert metadata["display_model"] == "Codex"
    assert metadata["route_type"] == "coding"
    assert "coding/debugging" in str(metadata["routing_reason"])
    assert metadata["fallback_used"] is False
    assert metadata["fallback_from"] is None
    assert metadata["forced_backend"] is False


def test_phase_a_metrics_record_fallback_and_forced_route_metadata(tmp_path: Path) -> None:
    fallback_path = tmp_path / "fallback"
    forced_path = tmp_path / "forced"
    fallback_path.mkdir()
    forced_path.mkdir()
    fallback_service = make_core_service(
        fallback_path,
        BackendRegistry(
            {
                "codex": FakeAdapter("codex", available=False),
                "claude-code": FakeAdapter("claude-code"),
                "ollama": FakeAdapter("ollama"),
            }
        ),
    )
    fallback_service.ask("Debug this repo test failure", backend="auto")
    fallback_metadata = fallback_service.metrics_list(limit=1)[0].metadata

    forced_service = make_core_service(
        forced_path,
        BackendRegistry({"ollama": FakeAdapter("ollama"), "codex": FakeAdapter("codex")}),
    )
    forced_service.ask("Debug this repo test failure", backend="ollama")
    forced_metadata = forced_service.metrics_list(limit=1)[0].metadata

    assert fallback_metadata["selected_backend"] == "claude-code"
    assert fallback_metadata["display_model"] == "Claude"
    assert fallback_metadata["fallback_used"] is True
    assert fallback_metadata["fallback_from"] == "codex"
    assert fallback_metadata["forced_backend"] is False
    assert forced_metadata["selected_backend"] == "ollama"
    assert forced_metadata["route_type"] == "forced"
    assert forced_metadata["forced_backend"] is True


def test_backend_ask_records_unexpected_adapter_exception(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"codex": RaisingAdapter("codex")}),
    )

    response = service.ask("Debug this repo", backend="codex")
    records = service.metrics_list(limit=1)

    assert not response.success
    assert "RuntimeError: boom" in (response.error_message or "")
    assert response.cost_type == BackendCostType.LOCAL
    assert records[0].backend == "codex"  # type: ignore[attr-defined]
    assert not records[0].success  # type: ignore[attr-defined]


def test_backend_ask_redacts_prompt_like_error_before_recording(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry({"codex": PromptEchoErrorAdapter("codex")}),
    )

    response = service.ask("ordinary prompt body", backend="codex")
    record = service.metrics_list(limit=1)[0]

    assert not response.success
    assert "ordinary prompt body" in (response.error_message or "")
    assert "ordinary prompt body" not in (record.error_message or "")
    assert "redacted" in (record.error_message or "")


def test_private_mode_blocks_sensitive_prompt_for_subscription_backend(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {"codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION)}
        ),
    )

    response = service.ask("Summarise my private medical notes", backend="codex")
    records = service.metrics_list(limit=1)

    assert not response.success
    assert "blocked by private mode" in (response.error_message or "")
    assert response.cost_type == BackendCostType.SUBSCRIPTION
    assert records[0].backend == "codex"  # type: ignore[attr-defined]
    assert not records[0].success  # type: ignore[attr-defined]
    assert "Private mode blocked subscription backend" in (
        records[0].routing_reason or ""  # type: ignore[attr-defined]
    )


def test_cli_ask_backend_flag_uses_core_service(monkeypatch, capsys) -> None:
    class FakeCoreService:
        def ask(  # noqa: ANN001
            self,
            prompt,
            *,
            backend,
            project,
            model,
            timeout_s,
            metadata=None,
            session_id=None,
            new_session=False,
        ):
            assert backend == "codex"
            assert prompt == "Debug this repo"
            assert session_id is None
            assert new_session is False
            return SwitchboardResponse(
                request_id="req_cli",
                backend="codex",
                content="done",
                stdout="done",
                latency_ms=5,
                success=True,
                routing_reason="User selected backend codex.",
                cost_type=BackendCostType.SUBSCRIPTION,
                estimated_cost_usd=0.0,
            )

    monkeypatch.setattr(
        "switchboard.cli.build_core_service",
        lambda **kwargs: FakeCoreService(),
    )

    ask_command(
        argparse.Namespace(
            prompt="Debug this repo",
            project=None,
            backend="codex",
            timeout=3,
            force_model=None,
            show_metadata=False,
            no_cache=True,
            show_prompt=False,
            strict=False,
            allow_cloud_once=False,
            override_reason=None,
            baseline=None,
        )
    )

    output = capsys.readouterr().out
    assert "done" in output
    assert "Backend: codex" in output
    assert "Routing: User selected backend codex." in output


def test_cli_ask_without_backend_defaults_to_core_auto(monkeypatch, capsys) -> None:
    class FakeCoreService:
        def ask(  # noqa: ANN001
            self,
            prompt,
            *,
            backend,
            project,
            model,
            timeout_s,
            metadata=None,
            session_id=None,
            new_session=False,
        ):
            assert backend is None
            assert prompt == "Debug this repo"
            assert model is None
            assert session_id is None
            assert new_session is False
            return SwitchboardResponse(
                request_id="req_cli_auto",
                backend="codex",
                content="done",
                stdout="done",
                latency_ms=5,
                success=True,
                routing_reason="Detected coding/debugging task; prefers Codex.",
                cost_type=BackendCostType.SUBSCRIPTION,
                estimated_cost_usd=0.0,
            )

    monkeypatch.setattr(
        "switchboard.cli.build_core_service",
        lambda **kwargs: FakeCoreService(),
    )

    ask_command(
        argparse.Namespace(
            prompt="Debug this repo",
            project=None,
            backend=None,
            timeout=3,
            force_model=None,
            show_metadata=False,
            no_cache=True,
            show_prompt=False,
            strict=False,
            allow_cloud_once=False,
            override_reason=None,
            baseline=None,
        )
    )

    output = capsys.readouterr().out
    assert "Calling backend auto" in output
    assert "done" in output
    assert "Backend: codex" in output
    assert "Routing: Detected coding/debugging task; prefers Codex." in output


def test_cli_route_uses_core_route_preview(monkeypatch, capsys) -> None:
    class FakeCoreService:
        def preview_route(  # noqa: ANN001
            self,
            prompt,
            *,
            backend,
            project,
            model,
            metadata,
        ):
            assert prompt == "Debug this repo"
            assert backend is None
            assert project is None
            assert model is None
            assert metadata == {"surface": "cli_route"}
            return BackendRouteDecision(
                backend="codex",
                selected_backend="codex",
                display_model="Codex",
                route_type="coding",
                routing_reason="Detected coding/debugging task; prefers Codex.",
            )

    monkeypatch.setattr(
        "switchboard.cli.build_core_service",
        lambda **kwargs: FakeCoreService(),
    )

    route_command(
        argparse.Namespace(
            prompt="Debug this repo",
            project=None,
            no_cache=True,
            show_prompt=False,
            debug=False,
            show_reasons=False,
            force_model=None,
            allow_cloud_once=False,
            override_reason=None,
            baseline=None,
        )
    )

    output = capsys.readouterr().out
    assert "Recommendation: Codex" in output
    assert "Backend: codex" in output
    assert "Route type: coding" in output
    assert "Routing: Detected coding/debugging task; prefers Codex." in output


@pytest.mark.parametrize(
    ("force_model", "expected_backend", "expected_model", "expected_next_step"),
    [
        (
            "claude-code",
            "claude-code",
            None,
            "Next step: switchboard ask --backend claude-code '<same prompt>'",
        ),
        (
            "ollama/qwen3:8b",
            "ollama",
            "ollama/qwen3:8b",
            (
                "Next step: switchboard ask --backend ollama "
                "--force-model ollama/qwen3:8b '<same prompt>'"
            ),
        ),
        (
            "ollama",
            "ollama",
            None,
            "Next step: switchboard ask --backend ollama '<same prompt>'",
        ),
    ],
)
def test_cli_route_next_step_preserves_forced_choice(
    force_model: str,
    expected_backend: str,
    expected_model: str | None,
    expected_next_step: str,
    monkeypatch,
    capsys,
) -> None:
    class FakeCoreService:
        def preview_route(  # noqa: ANN001
            self,
            prompt,
            *,
            backend,
            project,
            model,
            metadata,
        ):
            assert backend == expected_backend
            assert model == expected_model
            return BackendRouteDecision(
                backend=expected_backend,
                selected_backend=expected_backend,
                display_model=backend_display_name(expected_backend),
                route_type="forced",
                routing_reason=f"User selected backend {expected_backend}.",
                forced_backend=True,
            )

    monkeypatch.setattr(
        "switchboard.cli.build_core_service",
        lambda **kwargs: FakeCoreService(),
    )

    route_command(
        argparse.Namespace(
            prompt="Debug this repo",
            project=None,
            show_prompt=False,
            debug=False,
            show_reasons=False,
            force_model=force_model,
        )
    )

    output = capsys.readouterr().out
    assert expected_next_step in output


def test_cli_ask_force_model_ollama_forces_backend_without_model(monkeypatch, capsys) -> None:
    class FakeCoreService:
        def ask(  # noqa: ANN001
            self,
            prompt,
            *,
            backend,
            project,
            model,
            timeout_s,
            metadata=None,
            session_id=None,
            new_session=False,
        ):
            assert backend == "ollama"
            assert model is None
            return SwitchboardResponse(
                request_id="req_cli_ollama",
                backend="ollama",
                content="done",
                stdout="done",
                latency_ms=5,
                success=True,
                routing_reason="User selected backend ollama.",
                cost_type=BackendCostType.LOCAL,
                estimated_cost_usd=0.0,
            )

    monkeypatch.setattr(
        "switchboard.cli.build_core_service",
        lambda **kwargs: FakeCoreService(),
    )

    ask_command(
        argparse.Namespace(
            prompt="Debug this repo",
            project=None,
            backend=None,
            timeout=3,
            force_model="ollama",
            show_metadata=False,
        )
    )

    output = capsys.readouterr().out
    assert "Backend: ollama" in output
    assert "Success: True" in output


def test_cli_route_rejects_manual_catalogue_ids() -> None:
    with pytest.raises(SystemExit, match="manual subscription catalogue entry"):
        route_command(
            argparse.Namespace(
                prompt="Debug this repo",
                project=None,
                show_prompt=False,
                debug=False,
                show_reasons=False,
                force_model="manual/codex",
            )
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["route", "prompt", "--no-cache"],
        ["route", "prompt", "--allow-cloud-once"],
        ["route", "prompt", "--override-reason", "because"],
        ["route", "prompt", "--baseline", "manual/claude-web"],
        ["ask", "prompt", "--no-cache"],
        ["ask", "prompt", "--show-prompt"],
        ["ask", "prompt", "--strict"],
        ["ask", "prompt", "--allow-cloud-once"],
        ["ask", "prompt", "--override-reason", "because"],
        ["ask", "prompt", "--baseline", "manual/claude-web"],
    ],
)
def test_public_core_cli_rejects_personal_only_flags(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        make_parser().parse_args(argv)


def test_core_route_preview_and_auto_ask_choose_same_backend(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
                "claude-code": FakeAdapter(
                    "claude-code",
                    cost_type=BackendCostType.SUBSCRIPTION,
                ),
            }
        ),
    )

    prompt = "Debug this repo test failure"
    preview = service.preview_route(prompt)
    response = service.ask(prompt, backend=None)

    assert preview.backend == "codex"
    assert response.backend == preview.backend


def test_core_route_preview_keeps_sensitive_content_local(tmp_path: Path) -> None:
    service = make_core_service(
        tmp_path,
        BackendRegistry(
            {
                "ollama": FakeAdapter("ollama"),
                "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
                "claude-code": FakeAdapter(
                    "claude-code",
                    cost_type=BackendCostType.SUBSCRIPTION,
                ),
            }
        ),
    )

    decision = service.preview_route(
        "My SSN is 123-45-6789. Review this architecture for reliability."
    )

    assert decision.backend == "ollama"
    assert "Private mode detected sensitive content" in decision.routing_reason


def test_backends_command_prints_availability(monkeypatch, capsys) -> None:
    class FakeCoreService:
        def backends(self):  # noqa: ANN201
            return [
                BackendInfo(
                    name="codex",
                    available=True,
                    cost_type=BackendCostType.SUBSCRIPTION,
                    path="/usr/bin/codex",
                )
            ]

    monkeypatch.setattr("switchboard.cli.build_core_service", lambda: FakeCoreService())
    monkeypatch.setenv("SWITCHBOARD_WEB_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-web-key")
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "secret-finance-key")

    backends_command(argparse.Namespace(format="text"))

    output = capsys.readouterr().out
    assert "codex" in output
    assert "web-search" in output
    assert "finance" in output
    assert "secret-web-key" not in output
    assert "secret-finance-key" not in output


def test_doctor_command_prints_provider_status_without_keys(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'doctor.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    monkeypatch.setattr("switchboard.cli.get_settings", lambda: settings)
    monkeypatch.setenv("SWITCHBOARD_WEB_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-web-key")
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "secret-finance-key")
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.HTTPError("offline")),
    )

    doctor_command(argparse.Namespace())

    output = capsys.readouterr().out
    assert "Web search provider: brave configured" in output
    assert "Finance provider: alpha_vantage configured" in output
    assert "secret-web-key" not in output
    assert "secret-finance-key" not in output


def test_metrics_command_summary(monkeypatch, capsys) -> None:
    class FakeCoreService:
        def metrics_summary(self):  # noqa: ANN201
            return {"total_requests": 1, "requests_by_backend": {"codex": 1}}

    monkeypatch.setattr("switchboard.cli.build_core_service", lambda: FakeCoreService())

    metrics_command(argparse.Namespace(view="summary", last=20, format="text"))

    output = capsys.readouterr().out
    assert '"total_requests": 1' in output


def test_train_router_command_exits_cleanly_when_embedding_model_down(
    monkeypatch, tmp_path: Path
) -> None:
    # Tester finding (round 6, must-fix 2): `switchboard train-router` dumped a
    # raw traceback when Ollama was not running. The CLI must exit with one
    # helpful line instead.
    dataset = tmp_path / "router_dataset.jsonl"
    dataset.write_text('{"prompt": "hi", "label": "chat"}\n', encoding="utf-8")

    def unreachable(**kwargs):  # noqa: ANN003, ANN202
        raise EmbeddingUnavailableError("Embedding model unreachable: connect error")

    monkeypatch.setattr(
        "switchboard.training.train_router.train_from_files", unreachable
    )

    with pytest.raises(SystemExit) as excinfo:
        train_router_command(
            argparse.Namespace(
                dataset=str(dataset),
                output=str(tmp_path / "weights.json"),
                embedding_model="nomic-embed-text",
                external=False,
                augment=False,
                augment_limit=None,
            )
        )

    message = str(excinfo.value)
    assert "Embedding model unreachable" in message
    assert "ollama pull nomic-embed-text" in message
    assert "Traceback" not in message


def test_train_router_command_converts_raw_httpx_errors_too(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "router_dataset.jsonl"
    dataset.write_text('{"prompt": "hi", "label": "chat"}\n', encoding="utf-8")

    def unreachable(**kwargs):  # noqa: ANN003, ANN202
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "switchboard.training.train_router.train_from_files", unreachable
    )

    with pytest.raises(SystemExit) as excinfo:
        train_router_command(
            argparse.Namespace(
                dataset=str(dataset),
                output=str(tmp_path / "weights.json"),
                embedding_model="nomic-embed-text",
                external=False,
                augment=False,
                augment_limit=None,
            )
        )

    assert "ollama pull nomic-embed-text" in str(excinfo.value)


def test_train_dispatcher_command_exits_cleanly_when_embedding_model_down(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "switchboard.training.tool_dispatcher_dataset."
        "load_or_build_dispatcher_dataset",
        lambda path: [],
    )

    def unreachable(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise EmbeddingUnavailableError("Embedding model unreachable: connect error")

    monkeypatch.setattr("switchboard.training.train_router.train", unreachable)

    with pytest.raises(SystemExit) as excinfo:
        train_dispatcher_command(
            argparse.Namespace(
                dataset=str(tmp_path / "dispatcher.jsonl"),
                output=str(tmp_path / "weights.json"),
                embedding_model="nomic-embed-text",
            )
        )

    message = str(excinfo.value)
    assert "Embedding model unreachable" in message
    assert "ollama pull nomic-embed-text" in message


def test_train_sensitivity_command_exits_cleanly_when_embedding_model_down(
    monkeypatch, tmp_path: Path
) -> None:
    def unreachable(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise EmbeddingUnavailableError("Embedding model unreachable: connect error")

    monkeypatch.setattr("switchboard.training.train_router.train", unreachable)

    with pytest.raises(SystemExit) as excinfo:
        train_sensitivity_command(
            argparse.Namespace(
                output=str(tmp_path / "weights.json"),
                embedding_model="nomic-embed-text",
            )
        )

    message = str(excinfo.value)
    assert "Embedding model unreachable" in message
    assert "ollama pull nomic-embed-text" in message


def test_backend_error_hint_for_unsupported_codex_model() -> None:
    response = SwitchboardResponse(
        request_id="req_test",
        backend="codex",
        selected_model="gpt-5",
        success=False,
        error_message=(
            "The 'gpt-5' model is not supported when using Codex with a ChatGPT account."
        ),
        cost_type=BackendCostType.SUBSCRIPTION,
    )

    assert "Retry without --force-model" in (backend_error_hint(response) or "")
