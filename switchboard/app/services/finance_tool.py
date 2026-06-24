from __future__ import annotations

import re

from switchboard.app.models.capabilities import Capability, ToolResult
from switchboard.app.services.finance_providers import (
    FinanceProvider,
    StockQuote,
    default_finance_provider,
)
from switchboard.app.utils.redaction import sanitize_provider_error


class StockPriceTool:
    _COMPANY_TO_SYMBOL: dict[str, tuple[str, str]] = {
        "servicenow": ("NOW", "ServiceNow"),
        "service now": ("NOW", "ServiceNow"),
        "oracle": ("ORCL", "Oracle"),
        "nvidia": ("NVDA", "Nvidia"),
        "apple": ("AAPL", "Apple"),
        "microsoft": ("MSFT", "Microsoft"),
        "google": ("GOOGL", "Alphabet"),
        "alphabet": ("GOOGL", "Alphabet"),
        "amazon": ("AMZN", "Amazon"),
        "meta": ("META", "Meta"),
        "tesla": ("TSLA", "Tesla"),
        "netflix": ("NFLX", "Netflix"),
        "amd": ("AMD", "AMD"),
        "intel": ("INTC", "Intel"),
        "ibm": ("IBM", "IBM"),
        "salesforce": ("CRM", "Salesforce"),
        "adobe": ("ADBE", "Adobe"),
        "uber": ("UBER", "Uber"),
        "airbnb": ("ABNB", "Airbnb"),
        "palantir": ("PLTR", "Palantir"),
        "broadcom": ("AVGO", "Broadcom"),
        "qualcomm": ("QCOM", "Qualcomm"),
        "walmart": ("WMT", "Walmart"),
        "costco": ("COST", "Costco"),
        "jpmorgan": ("JPM", "JPMorgan Chase"),
        "goldman sachs": ("GS", "Goldman Sachs"),
        "visa": ("V", "Visa"),
        "mastercard": ("MA", "Mastercard"),
        "coca-cola": ("KO", "Coca-Cola"),
        "coca cola": ("KO", "Coca-Cola"),
        "pepsi": ("PEP", "PepsiCo"),
        "disney": ("DIS", "Disney"),
        "boeing": ("BA", "Boeing"),
        "berkshire": ("BRK-B", "Berkshire Hathaway"),
        "shopify": ("SHOP", "Shopify"),
        "spotify": ("SPOT", "Spotify"),
        "coinbase": ("COIN", "Coinbase"),
        "sony": ("SONY", "Sony"),
        "toyota": ("TM", "Toyota"),
        "sap": ("SAP", "SAP"),
        "infosys": ("INFY", "Infosys"),
        "wipro": ("WIT", "Wipro"),
        "hdfc bank": ("HDB", "HDFC Bank"),
        "icici": ("IBN", "ICICI Bank"),
        "reliance": ("RELIANCE.NS", "Reliance Industries"),
        "tata consultancy": ("TCS.NS", "Tata Consultancy Services"),
        "tcs": ("TCS.NS", "Tata Consultancy Services"),
    }
    _KNOWN_TICKERS: dict[str, str] = {
        "NOW": "ServiceNow",
        "ORCL": "Oracle",
        "NVDA": "Nvidia",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "GOOGL": "Alphabet",
    }

    def __init__(self, provider: FinanceProvider | None = None) -> None:
        self.provider = provider or default_finance_provider()

    def answer(self, prompt: str) -> ToolResult | None:
        symbol, company_name = self.resolve_symbol(prompt)
        if symbol is None:
            return ToolResult(
                tool_name="stock_price",
                capability=Capability.STOCK_PRICE,
                answer="",
                success=False,
                error="stock ticker could not be resolved",
                metadata={
                    "tool_available": self.provider.is_configured(),
                    "ticker_resolved": False,
                    "pass_through_to_model": True,
                },
            )
        if not self.provider.is_configured():
            return ToolResult(
                tool_name="stock_price",
                capability=Capability.STOCK_PRICE,
                answer="",
                success=False,
                error="finance provider is not configured",
                metadata={
                    "tool_available": False,
                    "ticker_resolved": True,
                    "resolved_symbol": symbol,
                    "resolved_company_name": company_name,
                    "pass_through_to_model": True,
                },
            )
        try:
            quote = self.provider.get_quote(symbol)
        except Exception as exc:
            return ToolResult(
                tool_name="stock_price",
                capability=Capability.STOCK_PRICE,
                answer="",
                success=False,
                error=sanitize_provider_error(str(exc), prompt=prompt, backend="finance"),
                metadata={
                    "tool_available": True,
                    "ticker_resolved": True,
                    "resolved_symbol": symbol,
                    "resolved_company_name": company_name,
                    "pass_through_to_model": True,
                    "finance_error": sanitize_provider_error(
                        type(exc).__name__,
                        prompt=prompt,
                        backend="finance",
                    ),
                },
            )
        return ToolResult(
            tool_name="stock_price",
            capability=Capability.STOCK_PRICE,
            answer=self._trusted_fact(quote=quote, fallback_company_name=company_name),
            display_model_or_label="Finance",
            metadata=self._quote_metadata(quote, fallback_company_name=company_name),
        )

    def status(self) -> ToolResult:
        provider_name = getattr(self.provider, "name", self.provider.__class__.__name__)
        if self.provider.is_configured():
            answer = f"Switchboard finance provider is configured: {provider_name}."
        else:
            answer = "Switchboard does not currently have a stock/finance provider configured."
        return ToolResult(
            tool_name="stock_price",
            capability=Capability.STOCK_PRICE,
            answer=answer,
            display_model_or_label="Finance",
            metadata={
                "tool_available": self.provider.is_configured(),
                "finance_source": provider_name,
                "pass_through_to_model": False,
            },
        )

    def resolve_symbol(self, prompt: str) -> tuple[str | None, str | None]:
        normalized = prompt.lower().replace("’", "'").replace("“", '"').replace("”", '"')
        for company, (symbol, display_name) in self._COMPANY_TO_SYMBOL.items():
            if re.search(rf"\b{re.escape(company)}\b", normalized):
                return symbol, display_name
        for symbol, company_name in self._KNOWN_TICKERS.items():
            if re.search(rf"\b{re.escape(symbol)}\b", prompt):
                return symbol, company_name
        # Generic fallback: an explicit uppercase ticker next to stock wording,
        # e.g. "INFY stock price" or "stock price of PLTR".
        explicit = re.search(
            r"\b([A-Z]{2,5})\b\s+(?:stock|shares?|quote)"
            r"|(?:stock|shares?|quote|price)\s+(?:of|for)\s+([A-Z]{2,5})\b",
            prompt,
        )
        if explicit:
            symbol = explicit.group(1) or explicit.group(2)
            return symbol, None
        return None, None

    def _trusted_fact(
        self,
        *,
        quote: StockQuote,
        fallback_company_name: str | None,
    ) -> str:
        parts = [
            "User asked for stock price.",
            (
                "Resolved company/ticker: "
                f"{quote.company_name or fallback_company_name or 'Unknown'} / {quote.symbol}."
            ),
        ]
        if quote.price is not None:
            currency = f" {quote.currency}" if quote.currency else ""
            parts.append(f"Latest available quote: {quote.price:.2f}{currency}.")
        if quote.exchange:
            parts.append(f"Exchange: {quote.exchange}.")
        parts.append(f"Source: {quote.source}.")
        if quote.timestamp:
            parts.append(f"Timestamp: {quote.timestamp.isoformat()}.")
        parts.append(f"Data may be delayed: {str(quote.is_delayed).lower()}.")
        return " ".join(parts)

    def _quote_metadata(
        self,
        quote: StockQuote,
        *,
        fallback_company_name: str | None,
    ) -> dict[str, object]:
        return {
            "tool_available": True,
            "ticker_resolved": True,
            "resolved_symbol": quote.symbol,
            "resolved_company_name": quote.company_name or fallback_company_name,
            "finance_source": quote.source,
            "quote_timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
            "quote_is_realtime": quote.is_realtime,
            "quote_is_delayed": quote.is_delayed,
            "pass_through_to_model": False,
        }
