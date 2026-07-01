from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel


class StockQuote(BaseModel):
    symbol: str
    company_name: str | None = None
    price: float | None = None
    currency: str | None = None
    exchange: str | None = None
    timestamp: datetime | None = None
    source: str
    is_realtime: bool = False
    is_delayed: bool = True
    raw_provider_status: str | None = None


class FinanceProvider(Protocol):
    def is_configured(self) -> bool:
        ...

    def get_quote(self, symbol: str) -> StockQuote:
        ...


class UnconfiguredFinanceProvider:
    name = "unconfigured"

    def is_configured(self) -> bool:
        return False

    def get_quote(self, symbol: str) -> StockQuote:
        raise RuntimeError("finance provider is not configured")


class MockFinanceProvider:
    name = "mock"

    def __init__(self, quotes: dict[str, StockQuote] | None = None) -> None:
        self.quotes = {symbol.upper(): quote for symbol, quote in (quotes or {}).items()}

    def is_configured(self) -> bool:
        return True

    def get_quote(self, symbol: str) -> StockQuote:
        normalized = symbol.upper()
        if normalized not in self.quotes:
            raise LookupError(f"mock quote not found for {normalized}")
        return self.quotes[normalized]


class AlphaVantageFinanceProvider:
    name = "alpha_vantage"

    def __init__(self, *, api_key: str | None = None, timeout_s: int = 10) -> None:
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.timeout_s = timeout_s

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_quote(self, symbol: str) -> StockQuote:
        if not self.api_key:
            raise RuntimeError("ALPHA_VANTAGE_API_KEY is not configured")
        response = httpx.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.upper(),
                "apikey": self.api_key,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        quote = payload.get("Global Quote") or {}
        price_text = quote.get("05. price")
        if not price_text:
            status = payload.get("Note") or payload.get("Information") or "quote unavailable"
            raise RuntimeError(str(status))
        latest_trading_day = quote.get("07. latest trading day")
        timestamp = None
        if latest_trading_day:
            timestamp = datetime.fromisoformat(latest_trading_day).replace(tzinfo=UTC)
        return StockQuote(
            symbol=(quote.get("01. symbol") or symbol).upper(),
            price=float(price_text),
            currency="USD",
            exchange=None,
            timestamp=timestamp,
            source="Alpha Vantage",
            is_realtime=False,
            is_delayed=True,
            raw_provider_status=None,
        )


class YFinanceProvider:
    name = "yfinance"

    def is_configured(self) -> bool:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False
        return True

    def get_quote(self, symbol: str) -> StockQuote:
        try:
            import yfinance
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is not installed; install switchboard-local[finance]"
            ) from exc
        ticker = yfinance.Ticker(symbol.upper())
        info = ticker.fast_info
        price = getattr(info, "last_price", None)
        if price is None and isinstance(info, dict):
            price = info.get("last_price") or info.get("lastPrice")
        if price is None:
            raise RuntimeError(f"quote unavailable for {symbol.upper()}")
        currency = getattr(info, "currency", None)
        if currency is None and isinstance(info, dict):
            currency = info.get("currency")
        exchange = getattr(info, "exchange", None)
        if exchange is None and isinstance(info, dict):
            exchange = info.get("exchange")
        company_name = None
        try:
            company_name = (ticker.info or {}).get("longName")
        except Exception:
            company_name = None
        return StockQuote(
            symbol=symbol.upper(),
            company_name=company_name,
            price=float(price),
            currency=currency,
            exchange=exchange,
            timestamp=datetime.now(UTC),
            source="yfinance",
            is_realtime=False,
            is_delayed=True,
        )


class YahooFinanceProvider:
    """Keyless quote provider backed by Yahoo Finance's public chart API.

    No API key required. Quotes may be delayed; this is informational data,
    not investment advice.
    """

    name = "yahoo"
    _BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(
        self,
        *,
        timeout_s: int = 10,
        fetch_json: Callable[[str], dict] | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self._fetch_json = fetch_json or self._http_fetch

    def is_configured(self) -> bool:
        return True

    def _http_fetch(self, symbol: str) -> dict:
        response = httpx.get(
            f"{self._BASE_URL}{quote_plus(symbol.upper())}",
            headers={"User-Agent": "Mozilla/5.0 (Switchboard local assistant)"},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def get_quote(self, symbol: str) -> StockQuote:
        payload = self._fetch_json(symbol)
        chart = payload.get("chart") or {}
        error = chart.get("error")
        results = chart.get("result") or []
        if error or not results:
            raise RuntimeError(str(error or f"quote unavailable for {symbol.upper()}"))
        meta = results[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        if price is None:
            raise RuntimeError(f"quote unavailable for {symbol.upper()}")
        market_time = meta.get("regularMarketTime")
        timestamp = (
            datetime.fromtimestamp(market_time, tz=UTC)
            if isinstance(market_time, int | float)
            else datetime.now(UTC)
        )
        return StockQuote(
            symbol=str(meta.get("symbol") or symbol).upper(),
            company_name=meta.get("shortName") or meta.get("longName"),
            price=float(price),
            currency=meta.get("currency"),
            exchange=meta.get("exchangeName") or meta.get("fullExchangeName"),
            timestamp=timestamp,
            source="Yahoo Finance",
            is_realtime=False,
            is_delayed=True,
        )


def finance_provider_by_name(name: str) -> FinanceProvider:
    normalized = (name or "").strip().lower()
    if normalized in {"", "none"}:
        return UnconfiguredFinanceProvider()
    if normalized in {"yahoo", "yahoo_finance"}:
        return YahooFinanceProvider()
    if normalized == "yfinance":
        return YFinanceProvider()
    if normalized in {"alpha_vantage", "alphavantage"}:
        return AlphaVantageFinanceProvider()
    return UnconfiguredFinanceProvider()


def default_finance_provider() -> FinanceProvider:
    return finance_provider_by_name(os.getenv("SWITCHBOARD_FINANCE_PROVIDER", ""))
