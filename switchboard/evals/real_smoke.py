from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings, get_settings
from switchboard.app.models.backends import SwitchboardRequest, SwitchboardResponse
from switchboard.app.models.telemetry import BackendMetricRead
from switchboard.app.services.container import ServiceContainer, build_container
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.app.utils.ids import new_request_id
from switchboard.app.utils.redaction import sanitize_provider_error
from switchboard.evals.datasets import real_smoke_cases
from switchboard.evals.reports import build_report
from switchboard.evals.types import EvalCase, EvalReport, EvalResult, EvalStatus

RegistryFactory = Callable[[ServiceContainer, Path], BackendRegistry]


class RealSmokeRunner:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cwd: Path | None = None,
        registry_factory: RegistryFactory | None = None,
        timeout_s: int = 90,
        case_timeouts: dict[str, int] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cwd = cwd or Path.cwd()
        self.registry_factory = registry_factory or self._default_registry
        self.timeout_s = timeout_s
        self.case_timeouts = case_timeouts or {}

    def run(
        self,
        *,
        limit: int | None = None,
        tags: set[str] | None = None,
    ) -> EvalReport:
        service = self._build_service()
        availability = self._backend_availability(service)
        cases = self._filter_cases(real_smoke_cases(), tags=tags)
        if limit is not None:
            cases = cases[:limit]
        results = [
            self._run_session_case(service, case, availability)
            if case.steps
            else self._run_single_case(service, case, availability)
            for case in cases
        ]
        return build_report(
            "real-smoke",
            results,
            mode="real",
            backend_availability=availability,
        )

    def _build_service(self) -> SwitchboardCoreService:
        engine = create_db_engine(self.settings.database_url)
        init_db(engine)
        container = build_container(self.settings, engine)
        # Evals verify Switchboard's default routing contract; user toggles
        # like claude_code_web_search must not change eval outcomes.
        container.personal_config.preferences.claude_code_web_search = False
        registry = self.registry_factory(container, self.cwd)
        return SwitchboardCoreService(
            registry=registry,
            metrics=container.backend_metrics_repository,
            container=container,
        )

    def _default_registry(self, container: ServiceContainer, cwd: Path) -> BackendRegistry:
        return BackendRegistry.default(container, cwd=cwd)

    def _backend_availability(self, service: SwitchboardCoreService) -> dict[str, bool]:
        return {backend.name: backend.available for backend in service.backends()}

    def _filter_cases(
        self,
        cases: list[EvalCase],
        *,
        tags: set[str] | None,
    ) -> list[EvalCase]:
        if not tags:
            return cases
        return [case for case in cases if tags.intersection(case.tags)]

    def _run_single_case(
        self,
        service: SwitchboardCoreService,
        case: EvalCase,
        availability: dict[str, bool],
    ) -> EvalResult:
        missing = self._missing_required_backends(case, availability)
        if missing:
            return self._not_verified(case, missing)
        timeout_s = self._timeout_for_case(case)
        route_diagnostics = self._route_diagnostics(service, case)
        started = time.perf_counter()
        response = service.ask(
            case.prompt,
            backend=case.backend,
            project="eval-real-smoke",
            timeout_s=timeout_s,
            metadata=self._metadata(case),
        )
        elapsed_seconds = round(time.perf_counter() - started, 3)
        metric = self._metric_for_response(service, response)
        result = self._result_from_response(
            case=case,
            response=response,
            metric=metric,
            timeout_s=timeout_s,
            elapsed_seconds=elapsed_seconds,
            route_diagnostics=route_diagnostics,
        )
        return self._score_structural(case, result, metric)

    def _run_session_case(
        self,
        service: SwitchboardCoreService,
        case: EvalCase,
        availability: dict[str, bool],
    ) -> EvalResult:
        missing = self._missing_required_backends(case, availability)
        if missing:
            return self._not_verified(case, missing)

        timeout_s = self._timeout_for_case(case)
        route_diagnostics = self._route_diagnostics_for_session(service, case)
        session_id: str | None = None
        responses: list[SwitchboardResponse] = []
        metrics: list[BackendMetricRead | None] = []
        started = time.perf_counter()
        for index, step in enumerate(case.steps):
            response = service.ask(
                step.prompt,
                backend=self._step_backend(step),
                project="eval-real-smoke",
                timeout_s=timeout_s,
                metadata={
                    **self._metadata(case),
                    "eval_step_index": index,
                    "eval_step_count": len(case.steps),
                },
                session_id=session_id,
                new_session=session_id is None,
            )
            responses.append(response)
            metrics.append(self._metric_for_response(service, response))
            session_id = response.session_id or session_id

        elapsed_seconds = round(time.perf_counter() - started, 3)
        final_response = responses[-1]
        final_metric = metrics[-1]
        result = self._result_from_response(
            case=case,
            response=final_response,
            metric=final_metric,
            timeout_s=timeout_s,
            elapsed_seconds=elapsed_seconds,
            route_diagnostics=route_diagnostics,
        )
        result.metrics_recorded = all(
            self._metric_has_eval_metadata(metric, case.case_id) for metric in metrics
        )
        result.notes.update(
            {
                "step_count": len(case.steps),
                "same_session": len({response.session_id for response in responses}) == 1,
                "context_message_count": self._int_value(
                    (final_metric.metadata if final_metric else {}).get("context_message_count")
                ),
                "context_injected": bool(
                    (final_metric.metadata if final_metric else {}).get("context_injected")
                ),
            }
        )
        return self._score_structural(case, result, final_metric)

    def _score_structural(
        self,
        case: EvalCase,
        result: EvalResult,
        metric: BackendMetricRead | None,
    ) -> EvalResult:
        failures: list[str] = []
        metadata = metric.metadata if metric is not None else {}
        display_model = metadata.get("display_model")
        route_selection_passed = result.notes.get("route_selection_passed")
        routing_failures: list[str] = []

        if route_selection_passed is False:
            routing_failures.append("route selection did not match expected backend")
        if not result.success and result.error_type != "timeout":
            failures.append("request did not complete successfully")
        if case.expected_backend and result.selected_backend != case.expected_backend:
            routing_failures.append(
                f"expected backend {case.expected_backend}, got {result.selected_backend}"
            )
        if case.expected_route_type and result.route_type != case.expected_route_type:
            routing_failures.append(
                f"expected route_type {case.expected_route_type}, got {result.route_type}"
            )
        if case.expected_tool and result.tool_name != case.expected_tool:
            failures.append(f"expected tool {case.expected_tool}, got {result.tool_name}")
        if (
            case.expected_capability
            and case.expected_capability not in result.detected_capabilities
        ):
            failures.append(
                f"expected capability {case.expected_capability}, "
                f"got {result.detected_capabilities}"
            )
        if not result.metrics_recorded:
            failures.append("eval metadata was not recorded in backend metrics")
        if (
            not display_model
            and result.selected_backend not in {"time", "switchboard"}
            and result.error_type != "timeout"
        ):
            failures.append("display model was not recorded")
        if result.success and not result.answer_preview:
            failures.append("answer was empty")
            result.error_type = result.error_type or "empty_response"
        if not case.should_call_model and result.selected_backend not in {"time", "switchboard"}:
            failures.append("tool-only case called a model backend")
        if case.steps:
            if result.notes.get("same_session") is not True:
                failures.append("session steps did not share one session")
            if result.notes.get("context_message_count", 0) <= 0:
                failures.append("later session step did not receive prior context")

        if routing_failures:
            failures = [*routing_failures, *failures]
            result.status = EvalStatus.FAIL
        elif result.error_type == "timeout":
            result.status = EvalStatus.TIMEOUT
            if not result.sanitized_error:
                result.sanitized_error = "backend timed out"
        else:
            result.status = EvalStatus.FAIL if failures else EvalStatus.PASS
        result.passed = result.status == EvalStatus.PASS
        result.failure_reason = "; ".join(failures) if failures else None
        return result

    def _result_from_response(
        self,
        *,
        case: EvalCase,
        response: SwitchboardResponse,
        metric: BackendMetricRead | None,
        timeout_s: int,
        elapsed_seconds: float,
        route_diagnostics: dict[str, object],
    ) -> EvalResult:
        metadata = metric.metadata if metric is not None else {}
        sanitized_error = sanitize_provider_error(
            response.error_message,
            prompt=case.prompt,
            backend=response.backend,
        )
        answer_preview = self._preview(response.content or "")
        is_model_backend = response.backend not in {"time", "switchboard"}
        is_timeout = self._is_timeout_error(response.error_message)
        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            name=case.name,
            passed=False,
            prompt=self._case_prompt(case),
            expected_backend=case.expected_backend,
            selected_backend=response.backend,
            expected_route_type=case.expected_route_type,
            route_type=self._string_or_none(metadata.get("route_type")),
            expected_tool=case.expected_tool,
            tool_name=self._string_or_none(metadata.get("tool_name")),
            expected_capability=case.expected_capability,
            primary_capability=self._string_or_none(metadata.get("primary_capability")),
            detected_capabilities=self._string_list(metadata.get("detected_capabilities")),
            should_call_model=case.should_call_model,
            model_called=is_model_backend,
            success=response.success,
            fallback_used=bool(metadata.get("fallback_used", False)),
            fallback_from=self._string_or_none(metadata.get("fallback_from")),
            metrics_recorded=self._metric_has_eval_metadata(metric, case.case_id),
            requested_backend_mode=case.backend or "auto",
            display_model=self._string_or_none(metadata.get("display_model"))
            or response.selected_model,
            routing_reason=metric.routing_reason if metric else response.routing_reason,
            timeout_seconds=timeout_s,
            elapsed_seconds=elapsed_seconds,
            error_type=self._error_type(response, answer_preview=answer_preview),
            sanitized_error=sanitized_error,
            answer_preview=answer_preview,
            process_started=is_model_backend,
            process_exited=not (is_timeout and is_model_backend),
            exit_code=response.exit_code,
            session_id=response.session_id,
            request_id=response.request_id,
            latency_ms=response.latency_ms,
            raw_answer_preview=self._preview(response.content or response.error_message or ""),
            notes=route_diagnostics,
        )

    def _not_verified(self, case: EvalCase, missing: list[str]) -> EvalResult:
        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            name=case.name,
            passed=False,
            prompt=self._case_prompt(case),
            status=EvalStatus.NOT_VERIFIED,
            failure_reason=f"required backend unavailable: {', '.join(missing)}",
            expected_backend=case.expected_backend,
            expected_route_type=case.expected_route_type,
            expected_tool=case.expected_tool,
            expected_capability=case.expected_capability,
            should_call_model=case.should_call_model,
            requested_backend_mode=case.backend or "auto",
            timeout_seconds=self._timeout_for_case(case),
            error_type="backend_unavailable",
            sanitized_error=f"required backend unavailable: {', '.join(missing)}",
            process_started=False,
            process_exited=False,
        )

    def _missing_required_backends(
        self,
        case: EvalCase,
        availability: dict[str, bool],
    ) -> list[str]:
        return [backend for backend in case.required_backends if not availability.get(backend)]

    def _metadata(self, case: EvalCase) -> dict[str, object]:
        return {
            "source": "eval_real_smoke",
            "eval_suite": "real-smoke",
            "eval_case_id": case.case_id,
            "eval_category": case.category,
            "eval_mock": False,
        }

    def _timeout_for_case(self, case: EvalCase) -> int:
        if case.case_id in self.case_timeouts:
            return self.case_timeouts[case.case_id]
        if case.steps:
            return max(self.timeout_s, 150)
        if "slow" in case.tags:
            return max(self.timeout_s, 120)
        if case.backend == "ollama" or case.expected_backend == "ollama":
            return min(self.timeout_s, 30) if self.timeout_s < 90 else 30
        return self.timeout_s

    def _route_diagnostics(
        self,
        service: SwitchboardCoreService,
        case: EvalCase,
    ) -> dict[str, object]:
        if case.expected_tool:
            return {"route_selection_status": "SKIPPED"}
        route_request = SwitchboardRequest(
            request_id=new_request_id(self.settings.request_id_prefix),
            prompt=case.prompt,
            project="eval-real-smoke",
            private_mode=service.container.personal_config.preferences.private_mode,
        )
        decision = service.route(route_request, forced_backend=case.backend)
        route_selection_passed = (
            not case.expected_backend or decision.backend == case.expected_backend
        )
        return {
            "route_selection_status": "PASS" if route_selection_passed else "FAIL",
            "route_selection_passed": route_selection_passed,
            "route_selected_backend": decision.backend,
            "route_expected_backend": case.expected_backend,
            "route_type": decision.route_type,
            "route_routing_reason": decision.routing_reason,
            "route_display_model": decision.display_model,
        }

    def _route_diagnostics_for_session(
        self,
        service: SwitchboardCoreService,
        case: EvalCase,
    ) -> dict[str, object]:
        if not case.steps:
            return self._route_diagnostics(service, case)
        final_step = case.steps[-1]
        final_case = EvalCase(
            case_id=case.case_id,
            category=case.category,
            name=case.name,
            prompt=final_step.prompt,
            expected_backend=final_step.expected_backend or case.expected_backend,
            expected_tool=final_step.expected_tool,
            backend=self._step_backend(final_step),
        )
        return self._route_diagnostics(service, final_case)

    def _is_timeout_error(self, error_message: str | None) -> bool:
        return bool(error_message and "timed out after" in error_message.lower())

    def _error_type(
        self,
        response: SwitchboardResponse,
        *,
        answer_preview: str,
    ) -> str | None:
        if response.success and answer_preview:
            return None
        if self._is_timeout_error(response.error_message):
            return "timeout"
        if response.success and not answer_preview:
            return "empty_response"
        if response.error_message:
            return "backend_execution_error"
        return None

    def _step_backend(self, step: Any) -> str | None:
        if step.expected_tool:
            return None
        if step.expected_backend in {"codex", "claude-code", "ollama"}:
            return step.expected_backend
        return None

    def _metric_for_response(
        self,
        service: SwitchboardCoreService,
        response: SwitchboardResponse,
    ) -> BackendMetricRead | None:
        for metric in service.metrics_list(limit=20):
            if metric.request_id == response.request_id:
                return metric
        return None

    def _metric_has_eval_metadata(
        self,
        metric: BackendMetricRead | None,
        case_id: str,
    ) -> bool:
        if metric is None:
            return False
        return metric.metadata.get("source") == "eval_real_smoke" and metric.metadata.get(
            "eval_case_id"
        ) == case_id

    def _string_or_none(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        return None

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []

    def _int_value(self, value: Any) -> int:
        return value if isinstance(value, int) else 0

    def _preview(self, value: str) -> str:
        return " ".join(value.split())[:180]

    def _case_prompt(self, case: EvalCase) -> str:
        if case.prompt:
            return case.prompt
        return "\n".join(step.prompt for step in case.steps)
