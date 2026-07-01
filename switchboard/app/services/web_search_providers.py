from __future__ import annotations

import os
from datetime import datetime
from typing import Protocol

import httpx
from pydantic import BaseModel


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str | None = None
    source: str | None = None
    published_at: datetime | None = None
    rank: int


class WebSearchProvider(Protocol):
    def is_configured(self) -> bool:
        ...

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        ...


class UnconfiguredWebSearchProvider:
    name = "unconfigured"

    def is_configured(self) -> bool:
        return False

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        raise RuntimeError("web search provider is not configured")


class MockWebSearchProvider:
    name = "mock"

    def __init__(self, results: dict[str, list[WebSearchResult]] | None = None) -> None:
        self.results = {query.lower(): values for query, values in (results or {}).items()}

    def is_configured(self) -> bool:
        return True

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        return self.results.get(query.lower(), [])[:max_results]


class BraveSearchProvider:
    name = "brave"

    def __init__(self, *, api_key: str | None = None, timeout_s: int = 10) -> None:
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.timeout_s = timeout_s

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured")
        response = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": self.api_key},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        items = (payload.get("web") or {}).get("results") or []
        results: list[WebSearchResult] = []
        for index, item in enumerate(items[:max_results], start=1):
            results.append(
                WebSearchResult(
                    title=item.get("title") or "Untitled result",
                    url=item.get("url") or "",
                    snippet=item.get("description"),
                    source="Brave Search",
                    rank=index,
                )
            )
        return results


def default_web_search_provider() -> WebSearchProvider:
    provider = os.getenv("SWITCHBOARD_WEB_PROVIDER", "").strip().lower()
    if provider in {"", "none"}:
        return UnconfiguredWebSearchProvider()
    if provider == "brave":
        return BraveSearchProvider()
    return UnconfiguredWebSearchProvider()
