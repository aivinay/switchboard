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
from switchboard.app.models.capabilities import Capability
from switchboard.app.services.capabilities import CapabilityDetector
from switchboard.app.services.container import build_container
from switchboard.app.services.finance_providers import (
    AlphaVantageFinanceProvider,
    MockFinanceProvider,
    StockQuote,
    UnconfiguredFinanceProvider,
    YFinanceProvider,
    default_finance_provider,
)
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.response_sanitizer import ResponseSanitizer
from switchboard.app.services.runtime_context import RuntimeContextProvider
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tools import TimeTool, ToolRegistry
from switchboard.app.services.web_search_providers import (
    BraveSearchProvider,
    MockWebSearchProvider,
    UnconfiguredWebSearchProvider,
    WebSearchResult,
    default_web_search_provider,
)
from switchboard.app.services.web_search_tool import WebSearchTool
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]
FIXED_UTC = datetime(2026, 6, 10, 14, 47, tzinfo=UTC)


class RecordingAdapter(AgentAdapter):
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        cost_type: BackendCostType = BackendCostType.LOCAL,
        content: str | None = None,
    ) -> None:
        self.name = name
        self.available = available
        self.cost_type = cost_type
        self.content = content
        self.calls: list[SwitchboardRequest] = []

    def is_available(self) -> bool:
        return self.available

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=self.available, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.calls.append(request)
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=self.content or f"{self.name} answered",
            selected_model=f"{self.name}/test",
            latency_ms=7,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


def fixed_runtime_context_provider() -> RuntimeContextProvider:
    return RuntimeContextProvider(
        local_timezone="America/New_York",
        clock=lambda: FIXED_UTC,
    )


def make_core_service(
    tmp_path: Path,
    adapters: dict[str, RecordingAdapter],
    *,
    tool_registry: ToolRegistry | None = None,
) -> SwitchboardCoreService:
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'capabilities.db'}",
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
        runtime_context_provider=fixed_runtime_context_provider(),
        tool_registry=tool_registry,
    )


def test_runtime_context_provider_returns_fixed_utc_and_local_time() -> None:
    context = fixed_runtime_context_provider().current()

    assert context.utc_datetime == FIXED_UTC
    assert context.local_timezone == "America/New_York"
    assert context.local_datetime.isoformat().startswith("2026-06-10T10:47:00")
    assert context.current_date == "June 10, 2026"
    assert "10:47 AM EDT" in context.human_local_time
    assert "2:47 PM UTC" in context.human_utc_time


def test_runtime_context_provider_uses_injected_clock_without_hardcoding_date() -> None:
    provider = RuntimeContextProvider(
        local_timezone="UTC",
        clock=lambda: datetime(2031, 1, 2, 3, 4, tzinfo=UTC),
    )

    context = provider.current()

    assert context.current_date == "January 2, 2031"
    assert context.local_iso.startswith("2031-01-02T03:04:00")


def test_brave_provider_config_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_WEB_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-key")

    provider = default_web_search_provider()

    assert isinstance(provider, BraveSearchProvider)
    assert provider.is_configured()


def test_brave_provider_not_configured_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_WEB_PROVIDER", "brave")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    provider = default_web_search_provider()

    assert isinstance(provider, BraveSearchProvider)
    assert not provider.is_configured()


def test_web_provider_none_is_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_WEB_PROVIDER", "none")

    assert isinstance(default_web_search_provider(), UnconfiguredWebSearchProvider)


def test_alpha_vantage_provider_config_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "secret-key")

    provider = default_finance_provider()

    assert isinstance(provider, AlphaVantageFinanceProvider)
    assert provider.is_configured()


def test_finance_provider_not_configured_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWITCHBOARD_FINANCE_PROVIDER", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    assert isinstance(default_finance_provider(), UnconfiguredFinanceProvider)


def test_yfinance_provider_selected_without_required_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "yfinance")

    provider = default_finance_provider()

    assert isinstance(provider, YFinanceProvider)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Time in India", Capability.CURRENT_TIME),
        ("What time is it?", Capability.CURRENT_TIME),
        ("current time in London", Capability.CURRENT_TIME),
        ("IST time now", Capability.CURRENT_TIME),
        ("UTC time", Capability.CURRENT_TIME),
        ("what date is it", Capability.CURRENT_DATE),
        ("What date is it today?", Capability.CURRENT_DATE),
        ("Today’s date", Capability.CURRENT_DATE),
        ("What day is today?", Capability.CURRENT_DATE),
        ("weather in India", Capability.WEATHER),
        ("current weather in Delhi", Capability.WEATHER),
        ("forecast for Mumbai", Capability.WEATHER),
        ("temperature in New York", Capability.WEATHER),
        ("latest OpenAI news", Capability.LATEST_INFO),
        ("current CEO of Microsoft", Capability.LATEST_INFO),
        ("search the web for LangChain release", Capability.WEB_SEARCH),
        ("look up online how to install Claude Code", Capability.WEB_SEARCH),
        ("stock price of ServiceNow", Capability.STOCK_PRICE),
        ("NOW stock price", Capability.STOCK_PRICE),
        ("current price of ORCL", Capability.STOCK_PRICE),
        ("what is NVDA trading at", Capability.STOCK_PRICE),
        ("current stock price", Capability.STOCK_PRICE),
        ("today’s market", Capability.LATEST_INFO),
        ("recent updates", Capability.LATEST_INFO),
        ("fix this bug", Capability.CODING),
        ("debug this failing test", Capability.CODING),
        ("refactor this function", Capability.CODING),
        ("analyze this repo", Capability.CODING),
        ("fix this test", Capability.CODING),
        ("review this architecture", Capability.REASONING),
        ("review this design", Capability.REASONING),
        ("think through tradeoffs", Capability.REASONING),
        ("explain the architecture", Capability.REASONING),
        ("create a paper plan", Capability.REASONING),
        ("answer locally", Capability.LOCAL_PRIVATE),
        ("private", Capability.LOCAL_PRIVATE),
        ("offline", Capability.LOCAL_PRIVATE),
        ("do not send to cloud", Capability.LOCAL_PRIVATE),
    ],
)
def test_capability_detector_flags_expected_categories(prompt: str, expected: Capability) -> None:
    detection = CapabilityDetector().detect(prompt)

    assert detection.has(expected)


@pytest.mark.parametrize(
    ("prompt", "expected_fragment"),
    [
        ("Time in India", "8:17 PM IST on June 10, 2026."),
        ("time in IST", "8:17 PM IST on June 10, 2026."),
        ("time in Asia/Kolkata", "8:17 PM IST on June 10, 2026."),
        ("time in New York", "10:47 AM EDT on June 10, 2026."),
        ("time in NYC", "10:47 AM EDT on June 10, 2026."),
        ("time in America/New_York", "10:47 AM EDT on June 10, 2026."),
        ("time in London", "3:47 PM BST on June 10, 2026."),
        ("time in Europe/London", "3:47 PM BST on June 10, 2026."),
        ("time in UTC", "2:47 PM UTC on June 10, 2026."),
        ("UTC time", "2:47 PM UTC on June 10, 2026."),
        ("what time is it", "10:47 AM EDT on June 10, 2026."),
    ],
)
def test_time_tool_computes_target_timezone(prompt: str, expected_fragment: str) -> None:
    context = fixed_runtime_context_provider().current()
    result = TimeTool().answer(
        prompt=prompt,
        capability=Capability.CURRENT_TIME,
        context=context,
    )

    assert result.success
    assert expected_fragment in result.answer


def test_time_tool_computes_current_date() -> None:
    context = fixed_runtime_context_provider().current()
    result = TimeTool().answer(
        prompt="what date is it",
        capability=Capability.CURRENT_DATE,
        context=context,
    )

    assert result.answer.startswith("Today is ")
    assert "June 10, 2026" in result.answer


@pytest.mark.parametrize(
    ("prompt", "symbol", "company"),
    [
        ("stock price of ServiceNow", "NOW", "ServiceNow"),
        ("ServiceNow stock price", "NOW", "ServiceNow"),
        ("Oracle stock", "ORCL", "Oracle"),
        ("what is NVDA trading at?", "NVDA", "Nvidia"),
        ("AAPL share price", "AAPL", "Apple"),
        ("NOW stock price", "NOW", "ServiceNow"),
    ],
)
def test_stock_price_tool_resolves_supported_tickers(
    prompt: str,
    symbol: str,
    company: str,
) -> None:
    assert StockPriceTool().resolve_symbol(prompt) == (symbol, company)


def test_time_question_uses_tool_grounding_and_forced_model(
    tmp_path: Path,
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

    response = service.ask("Time in India", backend="codex")

    assert response.success
    assert response.backend == "codex"
    assert response.selected_model == "codex/test"
    assert response.content == "codex answered"
    assert len(adapters["codex"].calls) == 1
    sent_prompt = adapters["codex"].calls[0].prompt
    assert "<trusted_facts>" in sent_prompt
    assert "The current time in India is 8:17 PM IST on June 10, 2026." in sent_prompt
    assert "<current_user_request>" in sent_prompt
    assert "Time in India" in sent_prompt

    record = service.metrics_list()[0]
    assert record.backend == "codex"
    assert isinstance(record.metadata["session_id"], str)
    assert isinstance(record.metadata["assistant_message_id"], str)
    assert record.metadata["detected_capabilities"] == ["current_time"]
    assert record.metadata["tool_used"] is True
    assert record.metadata["tool_name"] == "time"
    assert record.metadata["answered_by_tool"] is False
    assert record.metadata["grounded_by_tool"] is True
    assert record.metadata["model_called"] is True
    assert record.metadata["runtime_context_injected"] is False
    messages = service.context_store.list_messages(response.session_id or "")
    assert messages[-1].display_model == "Codex"


@pytest.mark.parametrize(
    ("prompt", "expected_capability"),
    [
        ("weather in Dubai", "weather"),
        ("ServiceNow stock price", "stock_price"),
        ("latest OpenAI news", "latest_info"),
    ],
)
def test_unconfigured_live_data_passes_through_to_selected_model(
    tmp_path: Path,
    prompt: str,
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

    assert response.success
    # Live-data questions without a provider route to the free local model
    # and carry an anti-fabrication instruction (dogfood regression).
    assert response.backend == "ollama"
    assert response.content == "ollama answered"
    assert len(adapters["ollama"].calls) == 1
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "Do not invent specific" in sent_prompt
    assert prompt in sent_prompt

    record = service.metrics_list()[0]
    assert expected_capability in record.metadata["detected_capabilities"]
    assert record.metadata["tool_used"] is False
    assert record.metadata["answered_by_tool"] is False
    assert record.metadata["grounded_by_tool"] is False
    assert record.metadata["tool_available"] is False
    assert record.metadata["pass_through_to_model"] is True
    assert record.metadata["model_called"] is True
    messages = service.context_store.list_messages(response.session_id or "")
    assert messages[-1].display_model == "Ollama"
    assert messages[-1].content == "ollama answered"


def test_explicit_weather_tool_status_can_use_status_grounding(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
            content="Switchboard does not currently have a weather provider configured.",
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("Does Switchboard have weather configured?", backend="auto")

    assert response.success
    assert response.backend == "ollama"
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "<trusted_facts>" in sent_prompt
    assert "Live weather is not configured yet." in sent_prompt
    record = service.metrics_list()[0]
    assert record.metadata["tool_used"] is True
    assert record.metadata["tool_name"] == "unsupported_live_data"
    assert record.metadata["grounded_by_tool"] is True
    assert record.metadata["pass_through_to_model"] is False


def test_stock_price_uses_mock_finance_provider_grounding(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
            content="ServiceNow (NOW) is trading at $112.45 USD.",
        ),
    }
    provider = MockFinanceProvider(
        {
            "NOW": StockQuote(
                symbol="NOW",
                company_name="ServiceNow",
                price=112.45,
                currency="USD",
                exchange="NYSE",
                timestamp=FIXED_UTC,
                source="Mock Finance",
                is_realtime=False,
                is_delayed=True,
            )
        }
    )
    service = make_core_service(
        tmp_path,
        adapters,
        tool_registry=ToolRegistry(stock_price_tool=StockPriceTool(provider)),
    )

    response = service.ask("stock price of ServiceNow", backend="auto")

    assert response.success
    assert response.backend == "ollama"
    # The model formats the answer; the trusted quote reaches it below.
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "<trusted_facts>" in sent_prompt
    assert "Resolved company/ticker: ServiceNow / NOW." in sent_prompt
    assert "Latest available quote: 112.45 USD." in sent_prompt
    assert "Source: Mock Finance." in sent_prompt
    record = service.metrics_list()[0]
    assert record.metadata["detected_capabilities"] == ["stock_price"]
    assert record.metadata["tool_used"] is True
    assert record.metadata["tool_name"] == "stock_price"
    assert record.metadata["grounded_by_tool"] is True
    assert record.metadata["ticker_resolved"] is True
    assert record.metadata["resolved_symbol"] == "NOW"
    assert record.metadata["resolved_company_name"] == "ServiceNow"
    assert record.metadata["finance_source"] == "Mock Finance"
    assert record.metadata["quote_is_delayed"] is True


class FailingFinanceProvider:
    name = "failing"

    def is_configured(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> StockQuote:
        raise RuntimeError(f"raw provider failure for {symbol}: secret details")


class RecordingWebSearchProvider(MockWebSearchProvider):
    def __init__(self) -> None:
        super().__init__(
            {
                "servicenow now stock price today": [
                    WebSearchResult(
                        title="ServiceNow quote",
                        url="https://example.test/now",
                        snippet="NOW traded at 112.45 USD.",
                        source="Mock Web",
                        rank=1,
                    )
                ],
                "dubai weather today": [
                    WebSearchResult(
                        title="Dubai weather",
                        url="https://example.test/dubai-weather",
                        snippet="Dubai is sunny.",
                        source="Mock Web",
                        rank=1,
                    )
                ],
                "latest openai news": [
                    WebSearchResult(
                        title="OpenAI news",
                        url="https://example.test/openai",
                        snippet="OpenAI published a mock update.",
                        source="Mock Web",
                        rank=1,
                    )
                ],
                "current CEO of Microsoft": [
                    WebSearchResult(
                        title="Microsoft CEO",
                        url="https://example.test/microsoft",
                        snippet="Satya Nadella is CEO.",
                        source="Mock Web",
                        rank=1,
                    )
                ],
                "search the web for LangChain release": [
                    WebSearchResult(
                        title="LangChain release",
                        url="https://example.test/langchain",
                        snippet="LangChain has a mock release.",
                        source="Mock Web",
                        rank=1,
                    )
                ],
            }
        )
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        self.queries.append(query)
        return super().search(query, max_results=max_results)


def test_stock_provider_failure_passes_through_without_raw_error(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(
        tmp_path,
        adapters,
        tool_registry=ToolRegistry(
            stock_price_tool=StockPriceTool(FailingFinanceProvider()),
        ),
    )

    response = service.ask("NOW stock price", backend="auto")

    assert response.success
    # Provider failure degrades to the local model with the honesty fact;
    # the raw provider error must never leak into the prompt.
    assert response.backend == "ollama"
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "raw provider failure" not in sent_prompt
    assert "Do not invent specific" in sent_prompt
    assert "NOW stock price" in sent_prompt
    record = service.metrics_list()[0]
    assert record.metadata["tool_used"] is False
    assert record.metadata["tool_available"] is True
    assert record.metadata["pass_through_to_model"] is True
    assert record.metadata["ticker_resolved"] is True
    assert record.metadata["resolved_symbol"] == "NOW"
    assert record.metadata["finance_error"] == "RuntimeError"


def test_explicit_stock_provider_status_can_use_status_grounding(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
            content="Switchboard does not currently have a stock provider configured.",
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("Does Switchboard have a stock provider configured?", backend="auto")

    assert response.success
    assert response.backend == "ollama"
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "<trusted_facts>" in sent_prompt
    assert "does not currently have a stock/finance provider configured" in sent_prompt
    record = service.metrics_list()[0]
    assert record.metadata["tool_used"] is True
    assert record.metadata["tool_name"] == "stock_price"
    assert record.metadata["tool_available"] is False


def test_time_tool_takes_priority_over_configured_web_search(tmp_path: Path) -> None:
    web_provider = RecordingWebSearchProvider()
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(
        tmp_path,
        adapters,
        tool_registry=ToolRegistry(web_search_tool=WebSearchTool(web_provider)),
    )

    response = service.ask("time in India", backend="auto")

    assert response.success
    assert web_provider.queries == []
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "The current time in India" in sent_prompt
    record = service.metrics_list()[0]
    assert record.metadata["tool_name"] == "time"
    assert record.metadata["web_search_used"] is False


@pytest.mark.parametrize(
    ("prompt", "expected_query", "expected_capability"),
    [
        ("stock price of ServiceNow", "ServiceNow NOW stock price today", "stock_price"),
        ("weather in Dubai", "Dubai weather today", "weather"),
        ("latest OpenAI news", "latest OpenAI news", "latest_info"),
        ("current CEO of Microsoft", "current CEO of Microsoft", "latest_info"),
        (
            "search the web for LangChain release",
            "search the web for LangChain release",
            "web_search",
        ),
    ],
)
def test_web_search_is_used_for_truth_prompts_when_configured(
    tmp_path: Path,
    prompt: str,
    expected_query: str,
    expected_capability: str,
) -> None:
    web_provider = RecordingWebSearchProvider()
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
            content="web grounded answer",
        ),
    }
    service = make_core_service(
        tmp_path,
        adapters,
        tool_registry=ToolRegistry(web_search_tool=WebSearchTool(web_provider)),
    )

    response = service.ask(prompt, backend="auto")

    assert response.success
    assert response.backend == "ollama"
    assert web_provider.queries == [expected_query]
    sent_prompt = adapters["ollama"].calls[0].prompt
    assert "<trusted_facts>" in sent_prompt
    assert "Web search query:" in sent_prompt
    record = service.metrics_list()[0]
    assert expected_capability in record.metadata["detected_capabilities"]
    assert record.metadata["tool_name"] == "web_search"
    assert record.metadata["web_search_used"] is True
    assert record.metadata["search_query"] == expected_query
    assert record.metadata["search_result_count"] == 1


@pytest.mark.parametrize(
    ("prompt", "expected_backend"),
    [
        ("hi", "ollama"),
        # Product decision 2026-06-12: unknown/personal prompts fail closed
        # to the free local model instead of a subscription backend.
        ("can I discuss love problems?", "ollama"),
        ("fix this bug", "codex"),
        ("review this architecture", "claude-code"),
        ("summarize this paragraph", "ollama"),
    ],
)
def test_non_truth_prompts_do_not_use_configured_web_search(
    tmp_path: Path,
    prompt: str,
    expected_backend: str,
) -> None:
    web_provider = RecordingWebSearchProvider()
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(
        tmp_path,
        adapters,
        tool_registry=ToolRegistry(web_search_tool=WebSearchTool(web_provider)),
    )

    response = service.ask(prompt, backend="auto")

    assert response.success
    assert response.backend == expected_backend
    assert web_provider.queries == []
    record = service.metrics_list()[0]
    assert record.metadata["web_search_used"] is False
    assert record.metadata["pass_through_to_model"] is False


@pytest.mark.parametrize(
    ("prompt", "expected_backend"),
    [
        ("Debug this repo", "codex"),
        ("Review this architecture", "claude-code"),
        ("answer locally", "ollama"),
    ],
)
def test_model_backed_requests_receive_clean_context_without_runtime_clock(
    tmp_path: Path,
    prompt: str,
    expected_backend: str,
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

    assert response.success
    assert response.backend == expected_backend
    assert len(adapters[expected_backend].calls) == 1
    sent_prompt = adapters[expected_backend].calls[0].prompt
    assert sent_prompt.startswith("You are replying to the user through Switchboard.")
    assert "Runtime context:" not in sent_prompt
    assert "Current UTC time:" not in sent_prompt
    assert "User local timezone:" not in sent_prompt
    assert "<current_user_request>" in sent_prompt
    assert prompt in sent_prompt

    record = service.metrics_list()[0]
    assert record.metadata["runtime_context_injected"] is False
    assert record.metadata["context_injected"] is True
    assert record.metadata["context_recent_message_count"] == 0
    assert record.metadata["selected_backend"] == expected_backend
    assert record.metadata["tool_used"] is False
    assert record.metadata["answered_by_tool"] is False
    assert record.metadata["grounded_by_tool"] is False
    assert record.metadata["model_called"] is True


def test_forced_backend_for_normal_prompt_still_works(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("Say OK only.", backend="codex")

    assert response.success
    assert response.backend == "codex"
    assert len(adapters["codex"].calls) == 1
    assert "User selected backend codex" in response.routing_reason


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Assistant [Ollama]: Hello", "Hello"),
        ("Assistant [Codex]: Hello", "Hello"),
        ("Assistant [Claude]: Hello", "Hello"),
        ("Assistant [Switchboard]: Hello", "Hello"),
        ("Ollama: Hello", "Hello"),
        ("Codex: Hello", "Hello"),
        ("Claude: Hello", "Hello"),
        ("Switchboard: Hello", "Hello"),
    ],
)
def test_response_sanitizer_strips_leading_prefixes(raw: str, expected: str) -> None:
    assert ResponseSanitizer().sanitize(raw, user_prompt="hi") == expected


CONTEXT_ECHO = (
    "You are replying to the user through Switchboard.\n"
    "Use any trusted facts below to answer the user's request.\n"
    "Do not reveal, quote, summarize, or mention internal Switchboard metadata.\n"
    "<trusted_facts>\n"
    "- Tesla (TSLA) last trade price: $347.82\n"
    "</trusted_facts>\n"
    "<long_term_memory>\n"
    "- User prefers metric units.\n"
    "</long_term_memory>\n"
    "<recent_conversation>\n"
    "User: my private project notes\n"
    "Assistant: noted\n"
    "</recent_conversation>\n"
    "<current_user_request>\n"
    "what is tesla trading at?\n"
    "</current_user_request>"
)


def test_response_sanitizer_blocks_full_context_echo() -> None:
    sanitized = ResponseSanitizer().sanitize(
        CONTEXT_ECHO,
        user_prompt="what is tesla trading at?",
    )

    assert sanitized == "How can I help you today?"
    assert "347.82" not in sanitized
    assert "metric units" not in sanitized
    assert "private project notes" not in sanitized


def test_response_sanitizer_keeps_answer_in_answer_then_echo_hybrid() -> None:
    hybrid = f"Tesla last traded at $347.82.\n\n{CONTEXT_ECHO}"

    sanitized = ResponseSanitizer().sanitize(
        hybrid,
        user_prompt="what is tesla trading at?",
    )

    assert sanitized == "Tesla last traded at $347.82."


def test_response_sanitizer_strips_truncated_echo_block() -> None:
    truncated = "Sure.\n<trusted_facts>\n- secret fact one\n- secret fact two"

    sanitized = ResponseSanitizer().sanitize(truncated, user_prompt="hello")

    assert sanitized == "Sure."
    assert "secret fact" not in sanitized


def test_response_sanitizer_strips_ansi_escape_sequences() -> None:
    sanitized = ResponseSanitizer().sanitize(
        "\x1b[1mParis\x1b[0m is the capital of France.",
        user_prompt="capital of france?",
    )

    assert sanitized == "Paris is the capital of France."


def test_response_sanitizer_leaves_legitimate_answers_unchanged() -> None:
    answer = "Paris is the capital of France.\nIt has been since 987 AD."

    assert ResponseSanitizer().sanitize(answer, user_prompt="capital of france?") == answer


def test_response_sanitizer_keeps_single_line_answer_with_internal_term() -> None:
    # A legitimate one-line answer mentioning "session" must not collapse to
    # the generic fallback (round-7 sanitizer minor).
    answer = "Your login session has expired; sign in again."

    assert ResponseSanitizer().sanitize(answer, user_prompt="why am i logged out?") == answer


@pytest.mark.parametrize("backend", ["auto", "ollama", "codex", "claude-code"])
def test_greeting_response_is_clean_for_auto_and_forced_backends(
    tmp_path: Path,
    backend: str,
) -> None:
    leaky = (
        "Assistant [Ollama]: Hi! This message is being routed through Switchboard.\n"
        "Hi! How can I help you today?"
    )
    adapters = {
        "ollama": RecordingAdapter("ollama", content=leaky),
        "codex": RecordingAdapter(
            "codex",
            cost_type=BackendCostType.SUBSCRIPTION,
            content="Codex: Hi! How can I help you today?",
        ),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
            content="Claude: Hi! How can I help you today?",
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("hi", backend=None if backend == "auto" else backend)

    assert response.success
    assert response.content == "Hi! How can I help you today?"
    lower = (response.content or "").lower()
    for forbidden in (
        "routed through switchboard",
        "runtime context",
        "utc time",
        "local time",
        "timezone",
        "logging",
        "logged",
        "session",
        "metrics",
        "routing reason",
        "backend",
    ):
        assert forbidden not in lower


@pytest.mark.parametrize("backend", ["ollama", "codex", "claude-code"])
def test_forced_model_with_time_grounding_uses_selected_backend(
    tmp_path: Path,
    backend: str,
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

    response = service.ask("time in India", backend=backend)

    assert response.success
    assert response.backend == backend
    assert len(adapters[backend].calls) == 1
    assert "The current time in India is 8:17 PM IST on June 10, 2026." in adapters[
        backend
    ].calls[0].prompt


def test_forced_unavailable_model_with_grounding_returns_clean_error(tmp_path: Path) -> None:
    adapters = {
        "ollama": RecordingAdapter("ollama"),
        "codex": RecordingAdapter(
            "codex",
            available=False,
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
        "claude-code": RecordingAdapter(
            "claude-code",
            cost_type=BackendCostType.SUBSCRIPTION,
        ),
    }
    service = make_core_service(tmp_path, adapters)

    response = service.ask("time in India", backend="codex")

    assert not response.success
    assert response.backend == "codex"
    assert response.content is None
    assert "Codex is unavailable" in (response.error_message or "")
    assert all(not adapter.calls for adapter in adapters.values())
