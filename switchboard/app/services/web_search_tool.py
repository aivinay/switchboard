from __future__ import annotations

from switchboard.app.models.capabilities import (
    Capability,
    CapabilityDetection,
    ToolResult,
)
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.web_search_providers import (
    WebSearchProvider,
    WebSearchResult,
    default_web_search_provider,
)
from switchboard.app.utils.redaction import sanitize_provider_error


class WebSearchTool:
    def __init__(self, provider: WebSearchProvider | None = None) -> None:
        self.provider = provider or default_web_search_provider()

    def is_configured(self) -> bool:
        return self.provider.is_configured()

    def answer(self, *, prompt: str, detection: CapabilityDetection) -> ToolResult:
        query = self.query_for(prompt=prompt, detection=detection)
        if not self.provider.is_configured():
            return ToolResult(
                tool_name="web_search",
                capability=Capability.WEB_SEARCH,
                answer="",
                success=False,
                error="web search provider is not configured",
                metadata={
                    "web_search_configured": False,
                    "web_search_used": False,
                    "search_query": query,
                    "search_result_count": 0,
                    "pass_through_to_model": True,
                },
            )
        try:
            results = self.provider.search(query, max_results=5)
        except Exception as exc:
            return ToolResult(
                tool_name="web_search",
                capability=Capability.WEB_SEARCH,
                answer="",
                success=False,
                error=sanitize_provider_error(str(exc), prompt=prompt, backend="web_search"),
                metadata={
                    "web_search_configured": True,
                    "web_search_used": False,
                    "search_query": query,
                    "search_result_count": 0,
                    "pass_through_to_model": True,
                    "web_search_error": sanitize_provider_error(
                        type(exc).__name__,
                        prompt=prompt,
                        backend="web_search",
                    ),
                },
            )
        if not results:
            return ToolResult(
                tool_name="web_search",
                capability=Capability.WEB_SEARCH,
                answer="",
                success=False,
                error="web search returned no results",
                metadata={
                    "web_search_configured": True,
                    "web_search_used": False,
                    "search_query": query,
                    "search_result_count": 0,
                    "pass_through_to_model": True,
                },
            )
        return ToolResult(
            tool_name="web_search",
            capability=Capability.WEB_SEARCH,
            answer=self._trusted_facts(query=query, results=results),
            display_model_or_label="Web",
            metadata={
                "web_search_configured": True,
                "web_search_used": True,
                "search_query": query,
                "search_result_count": len(results),
                "pass_through_to_model": False,
            },
        )

    def status(self) -> ToolResult:
        provider_name = getattr(self.provider, "name", self.provider.__class__.__name__)
        if self.provider.is_configured():
            answer = f"Switchboard web search provider is configured: {provider_name}."
        else:
            answer = "Switchboard does not currently have web search configured."
        return ToolResult(
            tool_name="web_search",
            capability=Capability.WEB_SEARCH,
            answer=answer,
            display_model_or_label="Web",
            metadata={
                "web_search_configured": self.provider.is_configured(),
                "web_search_used": False,
                "web_search_provider": provider_name,
                "pass_through_to_model": False,
            },
        )

    def query_for(self, *, prompt: str, detection: CapabilityDetection) -> str:
        text = " ".join(prompt.split())
        stock_symbol, company = StockPriceTool().resolve_symbol(prompt)
        if detection.has(Capability.STOCK_PRICE) and stock_symbol:
            name = company or stock_symbol
            return f"{name} {stock_symbol} stock price today"
        if detection.has(Capability.WEATHER):
            location = self._after_marker(text, ("weather in ", "weather for "))
            if location:
                return f"{location} weather today"
        return text

    def _after_marker(self, text: str, markers: tuple[str, ...]) -> str | None:
        lower = text.lower()
        for marker in markers:
            index = lower.find(marker)
            if index >= 0:
                return text[index + len(marker) :].strip(" ?.!")
        return None

    def _trusted_facts(self, *, query: str, results: list[WebSearchResult]) -> str:
        lines = [f"Web search query: {query}."]
        for result in results:
            snippet = f" Snippet: {result.snippet}" if result.snippet else ""
            published = (
                f" Published: {result.published_at.isoformat()}."
                if result.published_at
                else ""
            )
            lines.append(
                f"Result {result.rank}: {result.title}. URL: {result.url}."
                f"{snippet}{published} Source: {result.source or 'web search'}."
            )
        return "\n".join(lines)
