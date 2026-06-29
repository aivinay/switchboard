from __future__ import annotations

import os

from switchboard.app.services.finance_providers import finance_provider_by_name
from switchboard.app.services.news_tool import news_provider_by_name
from switchboard.app.services.web_search_providers import default_web_search_provider


def _requested_name(configured_name: str | None, env_name: str) -> str:
    configured = (configured_name or "").strip().lower()
    if configured:
        return configured
    return os.getenv(env_name, "none").strip().lower() or "none"


def web_provider_status() -> tuple[str, bool]:
    provider = default_web_search_provider()
    configured = provider.is_configured()
    requested = os.getenv("SWITCHBOARD_WEB_PROVIDER", "none").strip().lower() or "none"
    return requested if requested != "none" else getattr(provider, "name", "none"), configured


def finance_provider_status(configured_name: str | None = None) -> tuple[str, bool]:
    requested = _requested_name(configured_name, "SWITCHBOARD_FINANCE_PROVIDER")
    provider = finance_provider_by_name(requested)
    configured = provider.is_configured()
    return requested if requested != "none" else getattr(provider, "name", "none"), configured


def news_provider_status(configured_name: str | None = None) -> tuple[str, bool]:
    requested = _requested_name(configured_name, "SWITCHBOARD_NEWS_PROVIDER")
    provider = news_provider_by_name(requested)
    configured = provider.is_configured()
    return requested if requested != "none" else getattr(provider, "name", "none"), configured
