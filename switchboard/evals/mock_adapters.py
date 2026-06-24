from __future__ import annotations

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)

MODEL_BY_BACKEND = {
    "ollama": "ollama/llama3.2:3b",
    "codex": "codex/mock-default",
    "claude-code": "claude/mock-sonnet",
}

COST_BY_BACKEND = {
    "ollama": BackendCostType.LOCAL,
    "codex": BackendCostType.SUBSCRIPTION,
    "claude-code": BackendCostType.SUBSCRIPTION,
}


class MockAgentAdapter(AgentAdapter):
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        latency_ms: int = 3,
    ) -> None:
        self.name = name
        self.available = available
        self.latency_ms = latency_ms
        self.cost_type = COST_BY_BACKEND.get(name, BackendCostType.UNKNOWN)
        self.calls: list[SwitchboardRequest] = []

    def is_available(self) -> bool:
        return self.available

    def availability(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            available=self.available,
            cost_type=self.cost_type,
            path=f"mock://{self.name}" if self.available else None,
            details="deterministic eval adapter",
            warning=None if self.available else f"{self.name} unavailable in eval",
        )

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.calls.append(request)
        if not self.available:
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                success=False,
                latency_ms=self.latency_ms,
                error_message=f"{self.name} unavailable in eval",
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=self._answer(request.prompt),
            selected_model=MODEL_BY_BACKEND.get(self.name),
            stdout="",
            latency_ms=self.latency_ms,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )

    def _answer(self, prompt: str) -> str:
        text = prompt.lower()
        if "switchboard routes between codex, claude, and ollama" in text:
            return (
                "Mock review: Switchboard routes between Codex, Claude, and Ollama, "
                "which is the right CTO demo hook."
            )
        if "api.py due to none handling" in text:
            return (
                "Mock review: the api.py due to None handling hypothesis is plausible; "
                "verify it with a narrow regression test."
            )
        if "current time in india" in text:
            return "Mock answer: the current time in India came from trusted facts."
        if "current time in utc" in text:
            return "Mock answer: the current time in UTC came from trusted facts."
        if "current time in new york" in text:
            return "Mock answer: the current time in New York came from trusted facts."
        if "live weather is not configured" in text:
            return "Mock answer: live weather is not configured in Switchboard yet."
        if "live/latest information is not configured" in text:
            return "Mock answer: live/latest information is not configured yet."
        if "resolved company/ticker: servicenow / now" in text and "112.45" in text:
            return (
                "Mock answer: ServiceNow (NOW) is trading at $112.45 USD "
                "from Mock Finance; data may be delayed."
            )
        if "web search query:" in text:
            return "Mock answer: using trusted web search facts."
        if "<current_user_request> hi </current_user_request>" in " ".join(text.split()):
            return "Hi! How can I help you today?"
        if self.name == "codex":
            return "Mock Codex response: deterministic coding result."
        if self.name == "claude-code":
            return "Mock Claude response: deterministic reasoning result."
        if self.name == "ollama":
            return "Mock Ollama response: deterministic local result."
        return "Mock backend response."


def mock_registry(
    available_backends: dict[str, bool] | None = None,
) -> tuple[BackendRegistry, dict[str, MockAgentAdapter]]:
    availability = available_backends or {}
    adapters = {
        "ollama": MockAgentAdapter("ollama", available=availability.get("ollama", True)),
        "codex": MockAgentAdapter("codex", available=availability.get("codex", True)),
        "claude-code": MockAgentAdapter(
            "claude-code",
            available=availability.get("claude-code", True),
        ),
    }
    registry_adapters: dict[str, AgentAdapter] = dict(adapters)
    return BackendRegistry(registry_adapters), adapters
