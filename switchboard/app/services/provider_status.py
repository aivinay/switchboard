from __future__ import annotations

import os

from switchboard.app.services.finance_providers import default_finance_provider
from switchboard.app.services.web_search_providers import default_web_search_provider


def web_provider_status() -> tuple[str, bool]:
    provider = default_web_search_provider()
    configured = provider.is_configured()
    requested = os.getenv("SWITCHBOARD_WEB_PROVIDER", "none").strip().lower() or "none"
    return requested if requested != "none" else getattr(provider, "name", "none"), configured


def finance_provider_status() -> tuple[str, bool]:
    provider = default_finance_provider()
    configured = provider.is_configured()
    requested = os.getenv("SWITCHBOARD_FINANCE_PROVIDER", "none").strip().lower() or "none"
    return requested if requested != "none" else getattr(provider, "name", "none"), configured
