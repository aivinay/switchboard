from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings, get_settings
from switchboard.app.models.backends import SwitchboardResponse
from switchboard.app.models.telemetry import BackendMetricRead
from switchboard.app.services.container import build_container
from switchboard.app.services.finance_providers import MockFinanceProvider, StockQuote
from switchboard.app.services.finance_tool import StockPriceTool
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tools import ToolRegistry
from switchboard.app.services.web_search_providers import (
    MockWebSearchProvider,
    WebSearchResult,
)
from switchboard.app.services.web_search_tool import WebSearchTool
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.evals.datasets import cases_for_suite
from switchboard.evals.mock_adapters import MockAgentAdapter, mock_registry
from switchboard.evals.reports import build_report
from switchboard.evals.scorers import score_case
from switchboard.evals.types import EvalCase, EvalReport, EvalResult, EvalStep


class EvalRunner:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cwd: Path | None = None,
        mock: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.cwd = cwd or Path.cwd()
        self.mock = mock

    def list_cases(self, suite: str, *, limit: int | None = None) -> list[EvalCase]:
        cases = cases_for_suite(suite)
        if limit is not None:
            return cases[:limit]
        return cases

    def run(
        self,
        suite: str,
        *,
        backend: str = "auto",
        limit: int | None = None,
    ) -> EvalReport:
        cases = self.list_cases(suite, limit=limit)
        results = [self.run_case(case, suite=suite, backend=backend) for case in cases]
        return build_report(suite, results, mode="mock" if self.mock else "real")

    def run_case(self, case: EvalCase, *, suite: str, backend: str = "auto") -> EvalResult:
        if case.steps:
            return self._run_session_case(case, suite=suite)
        service, adapters = self._build_service(
            case.available_backends,
            mock_finance_provider=case.mock_finance_provider,
            mock_web_provider=case.mock_web_provider,
        )
        before_calls = self._call_count(adapters)
        response = service.ask(
            case.prompt,
            backend=self._backend_override(case.backend, backend),
            project="eval",
            metadata=self._metadata(case, suite=suite),
        )
        after_calls = self._call_count(adapters)
        metric = self._metric_for_response(service, response)
        result = self._result_from_response(
            case=case,
            response=response,
            metric=metric,
            model_called=after_calls > before_calls,
        )
        return score_case(case, result)

    def _run_session_case(self, case: EvalCase, *, suite: str) -> EvalResult:
        service, adapters = self._build_service(
            case.available_backends,
            mock_finance_provider=case.mock_finance_provider,
            mock_web_provider=case.mock_web_provider,
        )
        session_id: str | None = None
        responses: list[SwitchboardResponse] = []
        metrics: list[BackendMetricRead | None] = []
        model_called = False
        context_recall_required = False
        context_recall_observed = True

        for index, step in enumerate(case.steps):
            before_calls = self._call_count(adapters)
            response = service.ask(
                step.prompt,
                backend=self._step_backend(step),
                project="eval",
                metadata={
                    **self._metadata(case, suite=suite),
                    "eval_step_index": index,
                    "eval_step_count": len(case.steps),
                },
                session_id=session_id,
                new_session=session_id is None,
            )
            responses.append(response)
            metrics.append(self._metric_for_response(service, response))
            session_id = response.session_id or session_id
            model_called = model_called or self._call_count(adapters) > before_calls
            if step.expect_context_recall:
                context_recall_required = True
                observed = self._adapter_saw_context(
                    adapters=adapters,
                    backend=step.expected_backend,
                    phrase=step.expect_context_recall,
                )
                context_recall_observed = context_recall_observed and observed

        final_response = responses[-1]
        final_metric = metrics[-1]
        final_case = EvalCase(
            case_id=case.case_id,
            category=case.category,
            name=case.name,
            prompt=case.prompt,
            expected_backend=case.expected_backend,
            should_call_model=case.should_call_model,
        )
        result = self._result_from_response(
            case=final_case,
            response=final_response,
            metric=final_metric,
            model_called=model_called,
        )
        result.metrics_recorded = all(
            self._metric_has_eval_metadata(metric, case.case_id) for metric in metrics
        )
        result.notes.update(
            {
                "step_count": len(case.steps),
                "same_session": len({response.session_id for response in responses}) == 1,
                "context_recall_required": context_recall_required,
                "context_recall_observed": context_recall_observed,
            }
        )
        return score_case(case, result)

    def _build_service(
        self,
        available_backends: dict[str, bool] | None = None,
        *,
        mock_finance_provider: bool = False,
        mock_web_provider: bool = False,
    ) -> tuple[SwitchboardCoreService, dict[str, MockAgentAdapter]]:
        engine = create_db_engine(self.settings.database_url)
        init_db(engine)
        container = build_container(self.settings, engine)
        # Evals verify Switchboard's default routing contract; user toggles
        # like claude_code_web_search must not change eval outcomes.
        container.personal_config.preferences.claude_code_web_search = False
        adapters: dict[str, MockAgentAdapter] = {}
        if self.mock:
            registry, adapters = mock_registry(available_backends)
        else:
            registry = BackendRegistry.default(container, cwd=self.cwd)
        service = SwitchboardCoreService(
            registry=registry,
            metrics=container.backend_metrics_repository,
            container=container,
            tool_registry=self._tool_registry(
                mock_finance_provider=mock_finance_provider,
                mock_web_provider=mock_web_provider,
            ),
        )
        return service, adapters

    def _tool_registry(
        self,
        *,
        mock_finance_provider: bool,
        mock_web_provider: bool,
    ) -> ToolRegistry | None:
        if not mock_finance_provider and not mock_web_provider:
            return None
        stock_price_tool = None
        web_search_tool = None
        if mock_finance_provider:
            finance_provider = MockFinanceProvider(
                {
                    "NOW": StockQuote(
                        symbol="NOW",
                        company_name="ServiceNow",
                        price=112.45,
                        currency="USD",
                        exchange="NYSE",
                        timestamp=datetime(2026, 6, 10, 19, 30, tzinfo=UTC),
                        source="Mock Finance",
                        is_realtime=False,
                        is_delayed=True,
                    )
                }
            )
            stock_price_tool = StockPriceTool(finance_provider)
        if mock_web_provider:
            web_provider = MockWebSearchProvider(self._mock_web_results())
            web_search_tool = WebSearchTool(web_provider)
        return ToolRegistry(
            stock_price_tool=stock_price_tool,
            web_search_tool=web_search_tool,
        )

    def _mock_web_results(self) -> dict[str, list[WebSearchResult]]:
        return {
            "servicenow now stock price today": [
                WebSearchResult(
                    title="ServiceNow stock quote",
                    url="https://example.test/now",
                    snippet="NOW traded at 112.45 USD in mock search data.",
                    source="Mock Web",
                    rank=1,
                )
            ],
            "dubai weather today": [
                WebSearchResult(
                    title="Dubai weather",
                    url="https://example.test/dubai-weather",
                    snippet="Dubai weather is sunny in mock search data.",
                    source="Mock Web",
                    rank=1,
                )
            ],
            "latest openai news": [
                WebSearchResult(
                    title="OpenAI mock news",
                    url="https://example.test/openai-news",
                    snippet="OpenAI released a mock update.",
                    source="Mock Web",
                    rank=1,
                )
            ],
            "current CEO of Microsoft": [
                WebSearchResult(
                    title="Microsoft leadership",
                    url="https://example.test/microsoft-ceo",
                    snippet="Satya Nadella is listed as CEO in mock search data.",
                    source="Mock Web",
                    rank=1,
                )
            ],
            "search the web for LangChain release": [
                WebSearchResult(
                    title="LangChain mock release",
                    url="https://example.test/langchain-release",
                    snippet="LangChain has a mock release note.",
                    source="Mock Web",
                    rank=1,
                )
            ],
        }

    def _metadata(self, case: EvalCase, *, suite: str) -> dict[str, object]:
        return {
            "source": "eval",
            "eval_suite": suite,
            "eval_case_id": case.case_id,
            "eval_category": case.category,
            "eval_mock": self.mock,
        }

    def _backend_override(self, case_backend: str | None, command_backend: str) -> str | None:
        selected = case_backend or command_backend
        if selected == "auto":
            return None
        return selected

    def _step_backend(self, step: EvalStep) -> str | None:
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
        for metric in service.metrics_list(limit=10):
            if metric.request_id == response.request_id:
                return metric
        return None

    def _result_from_response(
        self,
        *,
        case: EvalCase,
        response: SwitchboardResponse,
        metric: BackendMetricRead | None,
        model_called: bool,
    ) -> EvalResult:
        metadata = metric.metadata if metric is not None else {}
        detected_capabilities = self._string_list(metadata.get("detected_capabilities"))
        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            name=case.name,
            passed=False,
            prompt=case.prompt,
            expected_backend=case.expected_backend,
            selected_backend=response.backend,
            expected_route_type=case.expected_route_type,
            route_type=self._string_or_none(metadata.get("route_type")),
            expected_tool=case.expected_tool,
            tool_name=self._string_or_none(metadata.get("tool_name")),
            expected_capability=case.expected_capability,
            primary_capability=self._string_or_none(metadata.get("primary_capability")),
            detected_capabilities=detected_capabilities,
            should_call_model=case.should_call_model,
            model_called=model_called,
            success=response.success,
            fallback_used=bool(metadata.get("fallback_used", False)),
            fallback_from=self._string_or_none(metadata.get("fallback_from")),
            metrics_recorded=self._metric_has_eval_metadata(metric, case.case_id),
            session_id=response.session_id,
            request_id=response.request_id,
            latency_ms=response.latency_ms,
            raw_answer_preview=self._preview(response.content or response.error_message or ""),
        )

    def _metric_has_eval_metadata(
        self,
        metric: BackendMetricRead | None,
        case_id: str,
    ) -> bool:
        if metric is None:
            return False
        return metric.metadata.get("source") == "eval" and metric.metadata.get(
            "eval_case_id"
        ) == case_id

    def _adapter_saw_context(
        self,
        *,
        adapters: dict[str, MockAgentAdapter],
        backend: str | None,
        phrase: str,
    ) -> bool:
        if backend is None:
            return False
        adapter = adapters.get(backend)
        if adapter is None or not adapter.calls:
            return False
        needle = phrase.lower()
        return needle in adapter.calls[-1].prompt.lower()

    def _call_count(self, adapters: dict[str, MockAgentAdapter]) -> int:
        return sum(len(adapter.calls) for adapter in adapters.values())

    def _string_or_none(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        return None

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        return []

    def _preview(self, value: str) -> str:
        return " ".join(value.split())[:180]
