from __future__ import annotations

import time
from pathlib import Path

from switchboard.app.core.config import Settings, get_settings
from switchboard.app.models.personal import PersonalConfig
from switchboard.app.services.container import build_container
from switchboard.app.services.core_factory import build_configured_core_service
from switchboard.app.services.finance_providers import (
    FinanceProvider,
    default_finance_provider,
    finance_provider_by_name,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.web_search_providers import default_web_search_provider
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.evals.reports import build_report
from switchboard.evals.types import EvalReport, EvalResult, EvalStatus


class RealProviderRunner:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cwd: Path | None = None,
        timeout_s: int = 120,
    ) -> None:
        self.settings = settings or get_settings()
        self.cwd = cwd or Path.cwd()
        self.timeout_s = timeout_s

    def run(self) -> EvalReport:
        results = [
            self._web_provider_status(),
            self._finance_provider_status("NOW"),
            self._finance_provider_status("ORCL"),
            self._stock_grounding(),
            self._web_grounding(),
        ]
        return build_report("real-providers", results, mode="real")

    def _service(self) -> SwitchboardCoreService:
        engine = create_db_engine(self.settings.database_url)
        init_db(engine)
        container = build_container(self.settings, engine)
        return build_configured_core_service(container, cwd=self.cwd)

    def _finance_provider(self) -> FinanceProvider:
        configured_name = self._configured_finance_provider_name()
        if configured_name:
            return finance_provider_by_name(configured_name)
        return default_finance_provider()

    def _configured_finance_provider_name(self) -> str:
        config = PersonalConfig.from_yaml(self.settings.personal_config_path)
        return config.preferences.finance_provider.strip()

    def _web_provider_status(self) -> EvalResult:
        provider = default_web_search_provider()
        case_id = "provider_web_brave"
        if not provider.is_configured():
            return self._not_verified(case_id, "Web provider", "web provider not configured")
        try:
            started = time.perf_counter()
            results = provider.search("latest OpenAI news", max_results=3)
            elapsed = round(time.perf_counter() - started, 3)
        except Exception as exc:
            return self._fail(case_id, "Web provider", f"{type(exc).__name__}: {exc}")
        if not results:
            return self._fail(case_id, "Web provider", "web provider returned no results")
        return self._pass(case_id, "Web provider", elapsed_seconds=elapsed)

    def _finance_provider_status(self, symbol: str) -> EvalResult:
        provider = self._finance_provider()
        case_id = f"provider_finance_{symbol.lower()}"
        if not provider.is_configured():
            return self._not_verified(
                case_id,
                f"Finance provider {symbol}",
                "finance provider not configured",
            )
        try:
            started = time.perf_counter()
            quote = provider.get_quote(symbol)
            elapsed = round(time.perf_counter() - started, 3)
        except Exception as exc:
            return self._fail(case_id, f"Finance provider {symbol}", f"{type(exc).__name__}: {exc}")
        if quote.price is None:
            return self._fail(case_id, f"Finance provider {symbol}", "quote price missing")
        return self._pass(case_id, f"Finance provider {symbol}", elapsed_seconds=elapsed)

    def _stock_grounding(self) -> EvalResult:
        finance = self._finance_provider()
        web = default_web_search_provider()
        case_id = "provider_stock_grounding"
        if not finance.is_configured() and not web.is_configured():
            return self._not_verified(
                case_id,
                "Stock grounding",
                "finance and web providers not configured",
            )
        service = self._service()
        started = time.perf_counter()
        response = service.ask(
            "stock price of ServiceNow",
            backend="auto",
            project="eval-real-providers",
            timeout_s=self.timeout_s,
            metadata={"source": "eval_real_providers", "eval_case_id": case_id},
        )
        elapsed = round(time.perf_counter() - started, 3)
        metric = self._metric(service, response.request_id)
        metadata = metric.metadata if metric else {}
        if not response.success or not response.content:
            return self._fail(case_id, "Stock grounding", response.error_message or "empty answer")
        if not metadata.get("model_called"):
            return self._fail(case_id, "Stock grounding", "selected model was not called")
        if finance.is_configured() and metadata.get("tool_name") != "stock_price":
            return self._fail(case_id, "Stock grounding", "finance tool was not used")
        if (
            not finance.is_configured()
            and web.is_configured()
            and metadata.get("tool_name") != "web_search"
        ):
            return self._fail(case_id, "Stock grounding", "web search was not used")
        return self._pass(case_id, "Stock grounding", elapsed_seconds=elapsed)

    def _web_grounding(self) -> EvalResult:
        web = default_web_search_provider()
        case_id = "provider_web_grounding"
        if not web.is_configured():
            return self._not_verified(case_id, "Web grounding", "web provider not configured")
        service = self._service()
        started = time.perf_counter()
        response = service.ask(
            "latest OpenAI news",
            backend="auto",
            project="eval-real-providers",
            timeout_s=self.timeout_s,
            metadata={"source": "eval_real_providers", "eval_case_id": case_id},
        )
        elapsed = round(time.perf_counter() - started, 3)
        metric = self._metric(service, response.request_id)
        metadata = metric.metadata if metric else {}
        if not response.success or not response.content:
            return self._fail(case_id, "Web grounding", response.error_message or "empty answer")
        if metadata.get("tool_name") != "web_search":
            return self._fail(case_id, "Web grounding", "web search tool was not used")
        if not metadata.get("model_called"):
            return self._fail(case_id, "Web grounding", "selected model was not called")
        return self._pass(case_id, "Web grounding", elapsed_seconds=elapsed)

    def _metric(self, service: SwitchboardCoreService, request_id: str):
        for metric in service.metrics_list(limit=20):
            if metric.request_id == request_id:
                return metric
        return None

    def _pass(self, case_id: str, name: str, *, elapsed_seconds: float | None = None) -> EvalResult:
        return EvalResult(
            case_id=case_id,
            category="real-providers",
            name=name,
            passed=True,
            status=EvalStatus.PASS,
            success=True,
            elapsed_seconds=elapsed_seconds,
        )

    def _fail(self, case_id: str, name: str, reason: str) -> EvalResult:
        return EvalResult(
            case_id=case_id,
            category="real-providers",
            name=name,
            passed=False,
            status=EvalStatus.FAIL,
            failure_reason=reason,
            success=False,
            sanitized_error=reason,
        )

    def _not_verified(self, case_id: str, name: str, reason: str) -> EvalResult:
        return EvalResult(
            case_id=case_id,
            category="real-providers",
            name=name,
            passed=False,
            status=EvalStatus.NOT_VERIFIED,
            failure_reason=reason,
            success=False,
            sanitized_error=reason,
        )
