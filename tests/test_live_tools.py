"""Tests for the keyless live-data tools: Yahoo Finance quotes and
Google News RSS headlines. No test touches the network."""

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
from switchboard.app.services.finance_providers import (
    UnconfiguredFinanceProvider,
    YahooFinanceProvider,
    finance_provider_by_name,
)
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.news_tool import (
    GoogleNewsRssProvider,
    MockNewsProvider,
    NewsHeadline,
    NewsTool,
    news_provider_by_name,
)
from switchboard.app.services.provider_status import (
    finance_provider_status,
    news_provider_status,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tools import ToolRegistry
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Yahoo Finance provider
# ---------------------------------------------------------------------------

YAHOO_PAYLOAD = {
    "chart": {
        "result": [
            {
                "meta": {
                    "symbol": "AAPL",
                    "shortName": "Apple Inc.",
                    "regularMarketPrice": 196.45,
                    "currency": "USD",
                    "exchangeName": "NMS",
                    "regularMarketTime": 1781200800,
                }
            }
        ],
        "error": None,
    }
}


def test_yahoo_provider_parses_quote() -> None:
    provider = YahooFinanceProvider(fetch_json=lambda symbol: YAHOO_PAYLOAD)
    quote = provider.get_quote("aapl")
    assert quote.symbol == "AAPL"
    assert quote.price == 196.45
    assert quote.currency == "USD"
    assert quote.company_name == "Apple Inc."
    assert quote.source == "Yahoo Finance"
    assert quote.is_delayed


def test_yahoo_provider_raises_on_error_payload() -> None:
    provider = YahooFinanceProvider(
        fetch_json=lambda symbol: {"chart": {"result": [], "error": {"code": "Not Found"}}}
    )
    with pytest.raises(RuntimeError):
        provider.get_quote("NOPE")


def test_finance_provider_factory() -> None:
    assert isinstance(finance_provider_by_name("yahoo"), YahooFinanceProvider)
    assert isinstance(finance_provider_by_name(""), UnconfiguredFinanceProvider)
    assert isinstance(finance_provider_by_name("bogus"), UnconfiguredFinanceProvider)


@pytest.mark.parametrize(
    ("prompt", "symbol"),
    [
        ("netflix stock price", "NFLX"),
        ("how is infosys stock doing", "INFY"),
        ("reliance stock price", "RELIANCE.NS"),
        ("stock price of PLTR", "PLTR"),
        ("INFY stock price", "INFY"),
    ],
)
def test_expanded_ticker_resolution(prompt: str, symbol: str) -> None:
    resolved, _ = StockPriceTool(UnconfiguredFinanceProvider()).resolve_symbol(prompt)
    assert resolved == symbol


# ---------------------------------------------------------------------------
# Google News RSS provider + NewsTool
# ---------------------------------------------------------------------------

RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Google News</title>
<item>
  <title>Markets rally on policy announcement - Example Times</title>
  <link>https://example.com/a</link>
  <pubDate>Thu, 11 Jun 2026 15:00:00 GMT</pubDate>
  <source url="https://example.com">Example Times</source>
</item>
<item>
  <title>Monsoon arrives early this year - Daily Sample</title>
  <link>https://example.com/b</link>
  <pubDate>Thu, 11 Jun 2026 14:00:00 GMT</pubDate>
  <source url="https://example.org">Daily Sample</source>
</item>
</channel></rss>"""


def test_google_news_rss_parses_headlines() -> None:
    fetched_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        fetched_urls.append(url)
        return RSS_SAMPLE

    provider = GoogleNewsRssProvider(fetch_text=fake_fetch)
    headlines = provider.headlines("india")

    assert len(headlines) == 2
    assert headlines[0].title.startswith("Markets rally")
    assert headlines[0].source == "Example Times"
    assert "q=india" in fetched_urls[0]


def test_google_news_rss_top_stories_without_query() -> None:
    provider = GoogleNewsRssProvider(fetch_text=lambda url: RSS_SAMPLE)
    assert len(provider.headlines("")) == 2


def test_news_tool_builds_trusted_fact_with_citations() -> None:
    provider = MockNewsProvider(
        items=[
            NewsHeadline(title="Headline one", source="Source A", published="today"),
            NewsHeadline(title="Headline two", source="Source B"),
        ]
    )
    result = NewsTool(provider).answer(prompt="give me latest news of india")

    assert result.success
    assert "Live headlines fetched at" in result.answer
    assert "1. Headline one (Source A, today)" in result.answer
    assert "do not invent additional news" in result.answer
    assert result.metadata["news_headline_count"] == 2
    # Filler words stripped from the query.
    assert provider.queries == ["india"]


def test_news_tool_failure_passes_through() -> None:
    class BoomProvider:
        name = "boom"

        def is_configured(self) -> bool:
            return True

        def headlines(self, query: str, *, limit: int = 5) -> list[NewsHeadline]:
            raise RuntimeError("rss fetch failed")

    result = NewsTool(BoomProvider()).answer(prompt="latest news")
    assert not result.success
    assert result.metadata["pass_through_to_model"] is True


def test_news_provider_factory() -> None:
    assert isinstance(news_provider_by_name("google_news_rss"), GoogleNewsRssProvider)
    assert news_provider_by_name("").is_configured() is False


def test_provider_status_empty_config_matches_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "yahoo")
    monkeypatch.setenv("SWITCHBOARD_NEWS_PROVIDER", "google_news_rss")

    assert finance_provider_status("") == ("yahoo", True)
    assert news_provider_status("") == ("google_news_rss", True)


def test_provider_status_explicit_none_disables_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "yahoo")
    monkeypatch.setenv("SWITCHBOARD_NEWS_PROVIDER", "google_news_rss")

    assert finance_provider_status("none")[1] is False
    assert news_provider_status("none")[1] is False


def test_tool_registry_availability_reflects_configured_live_tools() -> None:
    availability = ToolRegistry(
        news_tool=NewsTool(MockNewsProvider()),
        stock_price_tool=StockPriceTool(
            YahooFinanceProvider(fetch_json=lambda symbol: YAHOO_PAYLOAD)
        ),
    ).availability()

    assert availability["news_tool"] == "available"
    assert availability["live_latest_info_tool"] == "available"
    assert availability["stock_price_tool"] == "available"


# ---------------------------------------------------------------------------
# End to end: configured news/stock tools ground and route local
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
    tmp_path: Path, tool_registry: ToolRegistry
) -> tuple[SwitchboardCoreService, dict[str, FakeAdapter]]:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'live.db'}",
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
        tool_registry=tool_registry,
    )
    return service, adapters


def test_news_question_grounded_and_routed_local(tmp_path: Path) -> None:
    registry = ToolRegistry(
        news_tool=NewsTool(
            MockNewsProvider(items=[NewsHeadline(title="Big story", source="Wire")])
        )
    )
    service, adapters = make_service(tmp_path, registry)

    response = service.ask("give me latest news of india", new_session=True)

    assert response.success
    assert response.backend == "ollama"
    assert "deterministic tool grounded" in response.routing_reason.lower()
    prompt = adapters["ollama"].prompts[-1]
    assert "Big story" in prompt
    assert "do not invent additional news" in prompt


def test_stock_question_grounded_via_yahoo_and_routed_local(tmp_path: Path) -> None:
    registry = ToolRegistry(
        stock_price_tool=StockPriceTool(
            YahooFinanceProvider(fetch_json=lambda symbol: YAHOO_PAYLOAD)
        )
    )
    service, adapters = make_service(tmp_path, registry)

    response = service.ask("what is the stock price of Apple?", new_session=True)

    assert response.success
    assert response.backend == "ollama"
    assert "196.45" in adapters["ollama"].prompts[-1]
    assert "Yahoo Finance" in adapters["ollama"].prompts[-1]
    assert "whether the quote may be delayed" in adapters["ollama"].prompts[-1]


def test_news_provider_failure_falls_back_to_honest_local_answer(tmp_path: Path) -> None:
    class BoomProvider:
        name = "boom"

        def is_configured(self) -> bool:
            return True

        def headlines(self, query: str, *, limit: int = 5) -> list[NewsHeadline]:
            raise RuntimeError("offline")

    registry = ToolRegistry(news_tool=NewsTool(BoomProvider()))
    service, adapters = make_service(tmp_path, registry)

    response = service.ask("give me latest news of india", new_session=True)

    assert response.success
    assert response.backend == "ollama"
    # Anti-fabrication instruction applies when live grounding failed.
    assert "Do not invent specific" in adapters["ollama"].prompts[-1]
