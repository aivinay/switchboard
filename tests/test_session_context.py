from __future__ import annotations

from datetime import UTC, datetime
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
from switchboard.app.services.compression_layer import HeadroomCompressionLayer
from switchboard.app.services.container import build_container
from switchboard.app.services.finance_providers import MockFinanceProvider, StockQuote
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.runtime_context import RuntimeContextProvider
from switchboard.app.services.session_context import ContextBuilder, SessionManager
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tools import ToolRegistry
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.app.storage.repositories import ContextStore
from switchboard.app.utils.secret_patterns import redact_secrets

ROOT = Path(__file__).resolve().parents[1]
FIXED_UTC = datetime(2026, 6, 10, 14, 47, tzinfo=UTC)


class RecordingAdapter(AgentAdapter):
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
        self.calls: list[SwitchboardRequest] = []

    def is_available(self) -> bool:
        return self.available

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=self.available, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.calls.append(request)
        if not self.available:
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                success=False,
                error_message=f"{self.name} is unavailable",
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


def context_provider() -> RuntimeContextProvider:
    return RuntimeContextProvider(
        local_timezone="America/New_York",
        clock=lambda: FIXED_UTC,
    )


def context_store(tmp_path: Path) -> ContextStore:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    init_db(engine)
    return ContextStore(engine)


def make_core_service(
    tmp_path: Path,
    adapters: dict[str, RecordingAdapter],
) -> SwitchboardCoreService:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'core_sessions.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    return SwitchboardCoreService(
        registry=BackendRegistry(adapters),
        metrics=container.backend_metrics_repository,
        container=container,
        runtime_context_provider=context_provider(),
    )


def test_context_store_creates_appends_lists_and_handles_unknown_session(
    tmp_path: Path,
) -> None:
    store = context_store(tmp_path)

    session = store.create_session(title="Demo")
    user = store.append_message(session_id=session.session_id, role="user", content="Hello")
    assistant = store.append_message(
        session_id=session.session_id,
        role="assistant",
        content="Hi",
        display_model="Codex",
        backend="codex",
    )

    assert store.get_session(session.session_id) is not None
    assert store.get_session("missing") is None
    assert [message.message_id for message in store.list_messages(session.session_id)] == [
        user.message_id,
        assistant.message_id,
    ]
    assert store.list_messages("missing") == []
    recent = store.get_recent_messages(session.session_id, limit=1)
    assert recent[0].message_id == assistant.message_id

    with pytest.raises(ValueError, match="Unknown session_id"):
        store.append_message(session_id="missing", role="user", content="Nope")


def test_context_store_persists_messages_across_store_reload(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'durable_sessions.db'}"
    engine = create_db_engine(database_url)
    init_db(engine)
    store = ContextStore(engine)
    session = store.create_session()
    store.append_message(session_id=session.session_id, role="user", content="Persist me")

    reloaded_engine = create_db_engine(database_url)
    init_db(reloaded_engine)
    reloaded = ContextStore(reloaded_engine)

    assert reloaded.get_session(session.session_id) is not None
    assert reloaded.list_messages(session.session_id)[0].content == "Persist me"


def test_session_manager_reuses_or_creates_sessions(tmp_path: Path) -> None:
    manager = SessionManager(context_store(tmp_path))

    created = manager.resolve_session()
    reused = manager.resolve_session(session_id=created.session_id)
    explicit = manager.resolve_session(session_id="session_external")

    assert reused.session_id == created.session_id
    assert explicit.session_id == "session_external"


def test_context_builder_includes_recent_messages_without_runtime_or_model_labels(
    tmp_path: Path,
) -> None:
    store = context_store(tmp_path)
    session = store.create_session()
    store.append_message(session_id=session.session_id, role="user", content="Build the UI")
    store.append_message(
        session_id=session.session_id,
        role="assistant",
        content='{"raw":"json"}\nrequest_id: req_test\nImplemented it.',
        display_model="Codex",
        backend="codex",
        metadata={"backend": "codex", "metrics": True},
    )
    store.append_message(
        session_id=session.session_id,
        role="assistant",
        content="Traceback (most recent call last):\nSafe conclusion.",
        backend="claude-code",
    )
    updated_session = store.update_session_summary(
        session.session_id,
        "The user is building Switchboard.",
    )
    assert updated_session is not None

    result = ContextBuilder(max_recent_messages=2).build(
        session=updated_session,
        recent_messages=store.get_recent_messages(session.session_id, limit=10),
        runtime_context=context_provider().current(),
        current_request="Review what happened",
    )

    assert "Runtime context:" not in result.prompt
    assert "Current UTC time:" not in result.prompt
    assert "<recent_conversation>" in result.prompt
    assert "Summary: The user is building Switchboard." in result.prompt
    assert "Assistant: Implemented it. [JSON omitted from shared context.]" in result.prompt
    assert "Assistant: Safe conclusion. [Stack trace omitted from shared context.]" in result.prompt
    assert "Assistant [Codex]:" not in result.prompt
    assert "Assistant [Claude]:" not in result.prompt
    assert "claude-code" not in result.prompt
    assert "request_id:" not in result.prompt
    assert '"raw"' not in result.prompt
    assert "[JSON omitted from shared context.]" in result.prompt
    assert "[Stack trace omitted from shared context.]" in result.prompt
    assert result.recent_message_count == 2
    assert result.summary_used is True


def test_context_builder_redacts_obvious_secrets(tmp_path: Path) -> None:
    store = context_store(tmp_path)
    session = store.create_session()
    store.append_message(
        session_id=session.session_id,
        role="user",
        content=(
            "Use api_key=secret-value and Authorization: Bearer abc.def "
            "with sk-testsecret123456."
        ),
    )

    result = ContextBuilder().build(
        session=session,
        recent_messages=store.get_recent_messages(session.session_id, limit=10),
        runtime_context=context_provider().current(),
        current_request="Continue with password=hunter2",
    )

    assert "secret-value" not in result.prompt
    assert "abc.def" not in result.prompt
    assert "sk-testsecret123456" not in result.prompt
    assert "hunter2" not in result.prompt
    assert "[REDACTED" in result.prompt


def test_context_builder_respects_recent_message_limit(tmp_path: Path) -> None:
    store = context_store(tmp_path)
    session = store.create_session()
    for index in range(5):
        store.append_message(session_id=session.session_id, role="user", content=f"Message {index}")

    result = ContextBuilder(max_recent_messages=3).build(
        session=session,
        recent_messages=store.get_recent_messages(session.session_id, limit=10),
        runtime_context=context_provider().current(),
        current_request="Now answer",
    )

    assert "Message 0" not in result.prompt
    assert "Message 1" not in result.prompt
    assert "Message 2" in result.prompt
    assert "Message 4" in result.prompt
    assert result.recent_message_count == 3


def test_cross_model_continuity_across_forced_backends(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    first = service.ask("Remember this project goal: shared context.", backend="codex")
    second = service.ask(
        "What project goal did I just mention?",
        backend="claude-code",
        session_id=first.session_id,
    )
    third = service.ask(
        "Summarize our current project goal.",
        backend="ollama",
        session_id=first.session_id,
    )

    assert first.session_id == second.session_id == third.session_id
    claude_prompt = adapters["claude-code"].calls[0].prompt
    ollama_prompt = adapters["ollama"].calls[0].prompt
    assert "User: Remember this project goal: shared context." in claude_prompt
    assert "Assistant: codex answered" in claude_prompt
    assert "Assistant: claude-code answered" in ollama_prompt
    assert "Assistant [Claude]:" not in ollama_prompt

    messages = service.context_store.list_messages(first.session_id or "")
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_auto_mode_receives_shared_context(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    first = service.ask("Remember this codebase fact.", backend="codex")
    second = service.ask(
        "Review this architecture with that fact.",
        backend="auto",
        session_id=first.session_id,
    )

    assert second.backend == "claude-code"
    assert "Remember this codebase fact." in adapters["claude-code"].calls[0].prompt


@pytest.mark.parametrize(
    ("prompt", "expected_backend", "expected_route_type", "expected_capability"),
    [
        ("Fix this failing test", "codex", "coding", "coding"),
        ("Review this architecture", "claude-code", "reasoning", "reasoning"),
        ("Answer locally: summarize this", "ollama", "local", "local_private"),
    ],
)
def test_full_pipeline_model_backed_auto_cases_record_phase_metadata(
    tmp_path: Path,
    prompt: str,
    expected_backend: str,
    expected_route_type: str,
    expected_capability: str,
) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask(prompt, backend="auto")
    record = service.metrics_list(limit=1)[0]
    messages = service.context_store.list_messages(response.session_id or "")

    assert response.success
    assert response.backend == expected_backend
    assert len(adapters[expected_backend].calls) == 1
    assert "Runtime context:" not in adapters[expected_backend].calls[0].prompt
    assert record.metadata["selected_backend"] == expected_backend
    assert record.metadata["route_type"] == expected_route_type
    assert expected_capability in record.metadata["detected_capabilities"]
    assert record.metadata["tool_used"] is False
    assert record.metadata["answered_by_tool"] is False
    assert record.metadata["context_injected"] is True
    assert messages[-1].backend == expected_backend


def test_tool_answers_are_stored_and_available_to_next_backend(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    first = service.ask("Time in India", backend="auto")
    second = service.ask(
        "What time did you just tell me?",
        backend="codex",
        session_id=first.session_id,
    )

    assert first.backend == "ollama"
    assert second.backend == "codex"
    assert "Assistant: ollama answered" in adapters["codex"].calls[0].prompt
    assert "<trusted_facts>" not in adapters["codex"].calls[0].prompt


def test_retry_reuses_previous_stock_request_for_grounding(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)
    service.tool_registry = ToolRegistry(
        stock_price_tool=StockPriceTool(
            provider=MockFinanceProvider(
                {
                    "GOOGL": StockQuote(
                        symbol="GOOGL",
                        company_name="Alphabet",
                        price=356.38,
                        currency="USD",
                        exchange="Nasdaq",
                        source="mock",
                        is_delayed=True,
                    )
                }
            )
        )
    )

    first = service.ask("Stock price of Google", backend="auto")
    retry = service.ask("retry", backend="auto", session_id=first.session_id)

    assert retry.success
    retry_record = service.metrics_list(limit=1)[0]
    assert retry_record.metadata["followup_intent_reused"] is True
    assert retry_record.metadata["tool_name"] == "stock_price"
    assert retry_record.metadata["resolved_symbol"] == "GOOGL"
    assert retry_record.metadata["model_called"] is True

    retry_backend = retry.backend
    retry_prompt = adapters[retry_backend].calls[-1].prompt
    assert "<current_user_request>\nretry\n</current_user_request>" in retry_prompt
    assert "Latest available quote: 356.38 USD." in retry_prompt
    assert "Stock price of Google" in retry_prompt


def test_weather_unsupported_answer_is_stored(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("Weather in India", backend="auto")

    messages = service.context_store.list_messages(response.session_id or "")
    # Live-data without a provider now routes to the free local model.
    assert response.backend == "ollama"
    assert messages[-1].display_model == "Ollama"
    assert messages[-1].tool_name is None
    assert messages[-1].content == "ollama answered"


@pytest.mark.parametrize("backend", ["codex", "claude-code", "ollama"])
def test_forced_backend_receives_shared_context(tmp_path: Path, backend: str) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)
    first = service.ask("Remember this detail.", backend="ollama")

    service.ask("Use that detail.", backend=backend, session_id=first.session_id)

    assert "Remember this detail." in adapters[backend].calls[-1].prompt


def test_forced_unavailable_backend_records_session_and_returns_error(tmp_path: Path) -> None:
    adapters = {
        "codex": RecordingAdapter(
            "codex",
            available=False,
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("Say OK only.", backend="codex")

    assert not response.success
    assert response.session_id
    assert "unavailable" in (response.error_message or "")
    messages = service.context_store.list_messages(response.session_id)
    assert len(messages) == 1
    assert messages[0].role == "user"


def test_session_metrics_include_context_fields(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    first = service.ask("Remember this fact.", backend="codex")
    service.ask(
        "Use the fact in a design review.",
        backend="claude-code",
        session_id=first.session_id,
    )

    records = service.metrics_list(limit=2)
    latest = records[0]
    summary = service.metrics_summary()
    assert latest.metadata["session_id"] == first.session_id
    assert latest.metadata["context_injected"] is True
    assert latest.metadata["context_recent_message_count"] >= 2
    assert latest.metadata["selected_backend"] == "claude-code"
    assert latest.metadata["display_model"] == "Claude"
    assert summary["session_count"] == 1


PEM_SAMPLE = (
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEowIBAAKCAQEA7bq1Gz3kXhTestBodyLine1Abcdef\n"
    "Qw9yJX0mP1TestBodyLine2GhijklMnopqrstuvwxyz12\n"
    "-----END RSA PRIVATE KEY-----"
)
JWT_SAMPLE = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJVadQssw5c"
)


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("my access key id is AKIAIOSFODNN7EXAMPLE thanks", "AKIAIOSFODNN7EXAMPLE"),
        ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG", "wJalrXUtnFEMI"),
        ("set DATABASE_PASSWORD=correct-horse-battery in prod", "correct-horse-battery"),
        ("my password is hunter2", "hunter2"),
        ("the passphrase was tr0ub4dor3, do not lose it", "tr0ub4dor3"),
        (f"decode this for me: {JWT_SAMPLE}", JWT_SAMPLE),
        ("connect via postgres://switchboard:s3cretpw@db.internal:5432/app", "s3cretpw"),
        ("export GITHUB_TOKEN: ghp_abc123def456ghi789", "ghp_abc123def456ghi789"),
        ("curl -H 'x-api-key: live-9f8e7d6c5b4a'", "live-9f8e7d6c5b4a"),
        ("use sk-testsecret1234567890 for the API", "sk-testsecret1234567890"),
        ("Authorization: Bearer abc.def.ghi", "abc.def.ghi"),
        ("MY_SERVICE_CREDENTIALS=svc-user-09a8b7", "svc-user-09a8b7"),
    ],
)
def test_redact_secrets_covers_real_world_formats(text: str, secret: str) -> None:
    redacted = redact_secrets(text)

    assert secret not in redacted
    assert "[REDACTED" in redacted


@pytest.mark.parametrize(
    "text",
    [
        "the key is under the mat by the back door",
        "password hygiene tips for your team",
        "JWT is a token format commonly used on the web",
        "see https://example.com/docs?page=2 for details",
        "monkey=funny but turkey=dry",
        "sorted(items, key=len) is idiomatic Python",
    ],
)
def test_redact_secrets_leaves_benign_text_unchanged(text: str) -> None:
    assert redact_secrets(text) == text


def test_redact_secrets_removes_pem_blocks_including_truncated_ones() -> None:
    redacted = redact_secrets(f"here is the key\n{PEM_SAMPLE}\nplease rotate it")
    assert "MIIEow" not in redacted
    assert "TestBodyLine2" not in redacted
    assert "[REDACTED_PRIVATE_KEY]" in redacted
    assert "please rotate it" in redacted

    # Truncated paste: BEGIN marker without END still takes the body with it.
    dangling = redact_secrets("-----BEGIN PRIVATE KEY-----\nMIIEvBodyOnly123\nMoreBody456")
    assert "MIIEvBodyOnly123" not in dangling
    assert "[REDACTED_PRIVATE_KEY]" in dangling


def test_redact_secrets_preserves_url_username_and_host() -> None:
    redacted = redact_secrets("postgres://switchboard:s3cretpw@db.internal:5432/app")

    assert redacted == "postgres://switchboard:[REDACTED]@db.internal:5432/app"


def test_context_builder_redacts_multiline_pem_block_in_history(tmp_path: Path) -> None:
    store = context_store(tmp_path)
    session = store.create_session()
    store.append_message(
        session_id=session.session_id,
        role="user",
        content=f"Check this key:\n{PEM_SAMPLE}\nIs it valid?",
    )

    result = ContextBuilder().build(
        session=session,
        recent_messages=store.get_recent_messages(session.session_id, limit=10),
        runtime_context=context_provider().current(),
        current_request="So, can I commit it?",
    )

    assert "MIIEow" not in result.prompt
    assert "TestBodyLine2" not in result.prompt
    assert "[REDACTED_PRIVATE_KEY]" in result.prompt
    assert "Is it valid?" in result.prompt


def test_context_builder_redacts_extended_formats_in_current_request(tmp_path: Path) -> None:
    store = context_store(tmp_path)
    session = store.create_session()

    result = ContextBuilder().build(
        session=session,
        recent_messages=[],
        runtime_context=context_provider().current(),
        current_request=(
            "Why is AKIAIOSFODNN7EXAMPLE rejected?\n"
            f"The JWT was {JWT_SAMPLE}\n"
            "and DATABASE_PASSWORD=hunter2 is set."
        ),
    )

    assert "AKIAIOSFODNN7EXAMPLE" not in result.prompt
    assert JWT_SAMPLE not in result.prompt
    assert "hunter2" not in result.prompt
    assert "[REDACTED_AWS_KEY]" in result.prompt
    assert "[REDACTED_JWT]" in result.prompt


MULTILINE_CODE_REQUEST = (
    "Why does this print twice?\n"
    "def greet(names):\n"
    "    for name in names:\n"
    "        print(name)\n"
    "        print(name)"
)


def test_context_builder_preserves_newlines_in_current_request(tmp_path: Path) -> None:
    store = context_store(tmp_path)
    session = store.create_session()

    result = ContextBuilder().build(
        session=session,
        recent_messages=[],
        runtime_context=context_provider().current(),
        current_request=MULTILINE_CODE_REQUEST,
    )

    # Newlines AND indentation survive verbatim inside the request block.
    assert (
        f"<current_user_request>\n{MULTILINE_CODE_REQUEST}\n</current_user_request>"
        in result.prompt
    )


def test_ask_preserves_multiline_request_with_compression_enabled(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)
    service.compression = HeadroomCompressionLayer(threshold_tokens=100)

    first = service.ask("Remember the greet helper we discussed.", backend="codex")
    response = service.ask(
        MULTILINE_CODE_REQUEST,
        backend="codex",
        session_id=first.session_id,
    )

    assert response.success
    sent_prompt = adapters["codex"].calls[-1].prompt
    assert (
        f"<current_user_request>\n{MULTILINE_CODE_REQUEST}\n</current_user_request>"
        in sent_prompt
    )
    record = service.metrics_list(limit=1)[0]
    assert record.metadata["context_compression_enabled"] is True
