from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class BackendCostType(StrEnum):
    LOCAL = "local"
    SUBSCRIPTION = "subscription"
    API = "api"
    UNKNOWN = "unknown"


class SwitchboardRequest(BaseModel):
    request_id: str
    prompt: str
    project: str = "personal"
    model: str | None = None
    timeout_s: int = Field(default=120, ge=1)
    private_mode: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def prompt_char_count(self) -> int:
        return len(self.prompt)


class SwitchboardResponse(BaseModel):
    request_id: str
    backend: str
    session_id: str | None = None
    message_id: str | None = None
    content: str | None = None
    selected_model: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    latency_ms: int = 0
    success: bool
    error_message: str | None = None
    routing_reason: str = ""
    cost_type: BackendCostType = BackendCostType.UNKNOWN
    estimated_cost_usd: float | None = None


class BackendInfo(BaseModel):
    name: str
    available: bool
    cost_type: BackendCostType
    path: str | None = None
    details: str | None = None
    warning: str | None = None


DISPLAY_MODEL_BY_BACKEND = {
    "codex": "Codex",
    "claude-code": "Claude",
    "ollama": "Ollama",
    "switchboard": "Switchboard",
    "time": "Time",
}


def backend_display_name(backend: str) -> str:
    return DISPLAY_MODEL_BY_BACKEND.get(backend, "Switchboard")


class BackendRouteDecision(BaseModel):
    backend: str
    selected_backend: str | None = None
    display_model: str = "Switchboard"
    routing_reason: str
    route_type: str = "unknown"
    fallback_used: bool = False
    fallback_from: str | None = None
    forced_backend: bool = False
