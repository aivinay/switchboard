"""Live news grounding via RSS.

The default real provider is Google News RSS: free, keyless, and read-only.
Headlines are fetched deterministically and passed to the selected model as
trusted facts, so the model summarizes real headlines instead of inventing
them. This is informational data; sources should be cited in answers.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from pydantic import BaseModel

from switchboard.app.models.capabilities import Capability, ToolResult
from switchboard.app.utils.redaction import sanitize_provider_error


class NewsHeadline(BaseModel):
    title: str
    source: str | None = None
    published: str | None = None
    url: str | None = None


class NewsProvider(Protocol):
    name: str

    def is_configured(self) -> bool:
        ...

    def headlines(self, query: str, *, limit: int = 5) -> list[NewsHeadline]:
        ...


class UnconfiguredNewsProvider:
    name = "unconfigured"

    def is_configured(self) -> bool:
        return False

    def headlines(self, query: str, *, limit: int = 5) -> list[NewsHeadline]:
        raise RuntimeError("news provider is not configured")


class MockNewsProvider:
    name = "mock"

    def __init__(self, items: list[NewsHeadline] | None = None) -> None:
        self.items = items or []
        self.queries: list[str] = []

    def is_configured(self) -> bool:
        return True

    def headlines(self, query: str, *, limit: int = 5) -> list[NewsHeadline]:
        self.queries.append(query)
        return self.items[:limit]


class GoogleNewsRssProvider:
    """Keyless headlines from Google News RSS (search or top stories)."""

    name = "google_news_rss"
    _SEARCH_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    _TOP_URL = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

    def __init__(
        self,
        *,
        timeout_s: int = 10,
        fetch_text: Callable[[str], str] | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self._fetch_text = fetch_text or self._http_fetch

    def is_configured(self) -> bool:
        return True

    def _http_fetch(self, url: str) -> str:
        request = Request(  # noqa: S310 - fixed, well-known news endpoint.
            url,
            headers={"User-Agent": "Mozilla/5.0 (Switchboard local assistant)"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")

    def headlines(self, query: str, *, limit: int = 5) -> list[NewsHeadline]:
        url = (
            self._SEARCH_URL.format(query=quote_plus(query))
            if query.strip()
            else self._TOP_URL
        )
        root = ET.fromstring(self._fetch_text(url))
        items: list[NewsHeadline] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            source = (item.findtext("source") or "").strip() or None
            published = (item.findtext("pubDate") or "").strip() or None
            link = (item.findtext("link") or "").strip() or None
            items.append(
                NewsHeadline(title=title, source=source, published=published, url=link)
            )
            if len(items) >= limit:
                break
        return items


def news_provider_by_name(name: str) -> NewsProvider:
    normalized = (name or "").strip().lower()
    if normalized in {"", "none"}:
        return UnconfiguredNewsProvider()
    if normalized in {"google_news_rss", "google_news", "google"}:
        return GoogleNewsRssProvider()
    return UnconfiguredNewsProvider()


def default_news_provider() -> NewsProvider:
    return news_provider_by_name(os.getenv("SWITCHBOARD_NEWS_PROVIDER", ""))


_QUERY_FILLER = re.compile(
    r"\b(give me|tell me|show me|what is|what's|whats|the|latest|recent|current|"
    r"news|headlines|update|updates|about|on|of|for|today|now|please|any)\b",
    re.IGNORECASE,
)


class NewsTool:
    def __init__(self, provider: NewsProvider | None = None) -> None:
        self.provider = provider or default_news_provider()

    def is_configured(self) -> bool:
        return self.provider.is_configured()

    def query_for(self, prompt: str) -> str:
        cleaned = _QUERY_FILLER.sub(" ", prompt)
        cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
        return " ".join(cleaned.split())

    def answer(self, *, prompt: str) -> ToolResult:
        query = self.query_for(prompt)
        if not self.provider.is_configured():
            return ToolResult(
                tool_name="news",
                capability=Capability.LATEST_INFO,
                answer="",
                success=False,
                error="news provider is not configured",
                metadata={"news_configured": False, "pass_through_to_model": True},
            )
        try:
            headlines = self.provider.headlines(query, limit=5)
        except Exception as exc:
            return ToolResult(
                tool_name="news",
                capability=Capability.LATEST_INFO,
                answer="",
                success=False,
                error=sanitize_provider_error(str(exc), prompt=prompt, backend="news") or "",
                metadata={
                    "news_configured": True,
                    "news_provider": self.provider.name,
                    "pass_through_to_model": True,
                },
            )
        if not headlines:
            return ToolResult(
                tool_name="news",
                capability=Capability.LATEST_INFO,
                answer="",
                success=False,
                error="no headlines found",
                metadata={
                    "news_configured": True,
                    "news_provider": self.provider.name,
                    "news_query": query,
                    "pass_through_to_model": True,
                },
            )
        fetched_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            (
                f"Live headlines fetched at {fetched_at} from "
                f"{self.provider.name} (query: {query or 'top stories'}). "
                "Summarize only these; cite the sources; do not invent additional news."
            )
        ]
        for index, headline in enumerate(headlines, start=1):
            parts = [headline.title]
            if headline.source:
                parts.append(f"({headline.source}")
                parts[-1] += f", {headline.published})" if headline.published else ")"
            elif headline.published:
                parts.append(f"({headline.published})")
            lines.append(f"{index}. {' '.join(parts)}")
        return ToolResult(
            tool_name="news",
            capability=Capability.LATEST_INFO,
            answer="\n".join(lines),
            display_model_or_label="News",
            metadata={
                "news_configured": True,
                "news_provider": self.provider.name,
                "news_query": query,
                "news_headline_count": len(headlines),
                "pass_through_to_model": False,
            },
        )
