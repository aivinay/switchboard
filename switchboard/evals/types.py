from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EvalStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    NOT_VERIFIED = "NOT_VERIFIED"


@dataclass(frozen=True)
class EvalStep:
    prompt: str
    expected_backend: str | None = None
    expected_route_type: str | None = None
    expected_tool: str | None = None
    expected_capability: str | None = None
    should_call_model: bool = True
    expect_context_recall: str | None = None


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    name: str
    prompt: str
    expected_backend: str | None = None
    expected_route_type: str | None = None
    expected_tool: str | None = None
    expected_capability: str | None = None
    should_call_model: bool = True
    expected_success: bool = True
    expected_fallback_from: str | None = None
    backend: str | None = None
    available_backends: dict[str, bool] = field(default_factory=dict)
    mock_finance_provider: bool = False
    mock_web_provider: bool = False
    required_backends: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    steps: tuple[EvalStep, ...] = ()


@dataclass
class EvalResult:
    case_id: str
    category: str
    name: str
    passed: bool
    prompt: str = ""
    status: EvalStatus = EvalStatus.PASS
    failure_reason: str | None = None
    expected_backend: str | None = None
    selected_backend: str | None = None
    expected_route_type: str | None = None
    route_type: str | None = None
    expected_tool: str | None = None
    tool_name: str | None = None
    expected_capability: str | None = None
    primary_capability: str | None = None
    detected_capabilities: list[str] = field(default_factory=list)
    should_call_model: bool = True
    model_called: bool = False
    success: bool = False
    fallback_used: bool = False
    fallback_from: str | None = None
    metrics_recorded: bool = False
    requested_backend_mode: str = "auto"
    display_model: str | None = None
    routing_reason: str | None = None
    timeout_seconds: int | None = None
    elapsed_seconds: float | None = None
    error_type: str | None = None
    sanitized_error: str | None = None
    answer_preview: str = ""
    process_started: bool | None = None
    process_exited: bool | None = None
    exit_code: int | None = None
    session_id: str | None = None
    request_id: str | None = None
    latency_ms: int = 0
    raw_answer_preview: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalCategorySummary:
    total: int
    passed: int
    failed: int
    timed_out: int = 0
    skipped: int = 0
    not_verified: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class EvalReport:
    suite: str
    mode: str
    total: int
    passed: int
    failed: int
    timed_out: int
    skipped: int
    not_verified: int
    by_category: dict[str, EvalCategorySummary]
    failures: list[EvalResult]
    results: list[EvalResult]
    backend_availability: dict[str, bool] = field(default_factory=dict)
    average_latency_seconds_by_backend: dict[str, float] = field(default_factory=dict)
    timeouts_by_backend: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        denominator = self.total - self.skipped - self.not_verified
        if denominator <= 0:
            return 0.0
        return round(self.passed / denominator, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "mode": self.mode,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "timed_out": self.timed_out,
                "skipped": self.skipped,
                "not_verified": self.not_verified,
                "success_rate": self.success_rate,
                "pass_rate_excluding_not_verified": self.success_rate,
                "pass_rate_excluding_skipped_not_verified": self.success_rate,
            },
            "backend_availability": self.backend_availability,
            "average_latency_seconds_by_backend": self.average_latency_seconds_by_backend,
            "timeouts_by_backend": self.timeouts_by_backend,
            "by_category": {
                category: summary.to_dict()
                for category, summary in self.by_category.items()
            },
            "failures": [failure.to_dict() for failure in self.failures],
            "results": [result.to_dict() for result in self.results],
        }
