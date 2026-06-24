"""Status-intent detection: "is X configured?" questions about Switchboard.

These are meta-questions about the product itself ("do you have a weather
provider?"), answered from configuration rather than by a model or tool.
They were previously inline string piles in ToolRegistry; consolidated here
so the policy surface stays auditable in one place.

Behavior contract (unchanged from the inline version):

- ``asks_web_status``: any web-search-configuration phrase matches on its own.
- ``asks_tool_status``: requires a product-context word (``switchboard``,
  ``provider``, ``tool``, ``configured``, ``available``) AND a tool-status
  phrase, so ordinary prompts containing "weather" or "stock" never match.
"""

from __future__ import annotations

WEB_STATUS_MARKERS = (
    "web search configured",
    "web provider",
    "live search enabled",
    "search provider",
)

_PRODUCT_CONTEXT_MARKERS = ("provider", "tool", "configured", "available")

TOOL_STATUS_MARKERS = (
    "weather configured",
    "weather provider",
    "stock price provider",
    "stock provider",
    "finance provider",
    "stock lookup configured",
    "live-data tools",
    "live data tools",
    "tools are configured",
    "provider configured",
    "providers configured",
    "have weather",
    "have a stock",
    "stock lookup",
)


def asks_web_status(prompt: str) -> bool:
    text = prompt.lower()
    return any(marker in text for marker in WEB_STATUS_MARKERS)


def asks_tool_status(prompt: str) -> bool:
    text = prompt.lower()
    if "switchboard" not in text and not any(
        marker in text for marker in _PRODUCT_CONTEXT_MARKERS
    ):
        return False
    return any(marker in text for marker in TOOL_STATUS_MARKERS)
