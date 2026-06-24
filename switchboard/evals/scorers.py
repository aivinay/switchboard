from __future__ import annotations

from switchboard.evals.types import EvalCase, EvalResult, EvalStatus


def score_case(case: EvalCase, result: EvalResult) -> EvalResult:
    failures: list[str] = []

    if result.success != case.expected_success:
        failures.append(f"expected success={case.expected_success}, got {result.success}")
    if case.expected_backend and result.selected_backend != case.expected_backend:
        failures.append(
            f"expected backend {case.expected_backend}, got {result.selected_backend}"
        )
    if case.expected_route_type and result.route_type != case.expected_route_type:
        failures.append(
            f"expected route_type {case.expected_route_type}, got {result.route_type}"
        )
    if case.expected_tool and result.tool_name != case.expected_tool:
        failures.append(f"expected tool {case.expected_tool}, got {result.tool_name}")
    if case.expected_capability and case.expected_capability not in result.detected_capabilities:
        failures.append(
            "expected capability "
            f"{case.expected_capability}, got {result.detected_capabilities}"
        )
    if result.model_called != case.should_call_model:
        failures.append(
            f"expected model_called={case.should_call_model}, got {result.model_called}"
        )
    if case.expected_fallback_from and result.fallback_from != case.expected_fallback_from:
        failures.append(
            f"expected fallback_from {case.expected_fallback_from}, got {result.fallback_from}"
        )
    if not result.metrics_recorded:
        failures.append("eval metadata was not recorded in backend metrics")
    if case.category == "session":
        if result.notes.get("same_session") is not True:
            failures.append("session steps did not share one session")
        if result.notes.get("context_recall_required") and not result.notes.get(
            "context_recall_observed"
        ):
            failures.append("later backend call did not receive expected session context")

    result.passed = not failures
    result.status = EvalStatus.PASS if result.passed else EvalStatus.FAIL
    result.failure_reason = "; ".join(failures) if failures else None
    return result
