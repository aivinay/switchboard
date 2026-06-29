from __future__ import annotations

import argparse
import json
from pathlib import Path

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.services.finance_providers import StockQuote, UnconfiguredFinanceProvider
from switchboard.cli import (
    eval_command,
    eval_real_providers_command,
    eval_real_smoke_command,
)
from switchboard.evals.datasets import cases_for_suite
from switchboard.evals.mock_adapters import MockAgentAdapter
from switchboard.evals.real_providers import RealProviderRunner
from switchboard.evals.real_smoke import RealSmokeRunner
from switchboard.evals.reports import build_report, report_to_json
from switchboard.evals.runner import EvalRunner
from switchboard.evals.scorers import score_case
from switchboard.evals.types import EvalCase, EvalResult, EvalStatus

ROOT = Path(__file__).resolve().parents[1]


def eval_settings(tmp_path: Path, *, personal_config_path: Path | None = None) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'evals.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(personal_config_path or ROOT / "config" / "personal.yaml"),
    )


def live_provider_config(
    tmp_path: Path,
    *,
    finance_provider: str = "none",
    news_provider: str = "none",
) -> Path:
    personal_config = tmp_path / "personal_eval.yaml"
    personal_config.write_text(
        f"""
preferences:
  router_mode: "rules"
  tool_dispatcher_enabled: false
  sensitivity_escalator_enabled: false
  semantic_memory_enabled: false
  finance_provider: "{finance_provider}"
  news_provider: "{news_provider}"
""",
        encoding="utf-8",
    )
    return personal_config


def test_eval_datasets_have_unique_case_ids() -> None:
    cases = cases_for_suite("all")
    case_ids = [case.case_id for case in cases]

    assert len(cases) >= 45
    assert len(case_ids) == len(set(case_ids))
    assert {case.category for case in cases} == {"routing", "tools", "session"}


def test_expanded_mock_eval_case_counts() -> None:
    assert len(cases_for_suite("routing")) >= 20
    assert len(cases_for_suite("tools")) >= 15
    assert len(cases_for_suite("session")) >= 10


def test_routing_eval_passes_with_mock_backends(tmp_path: Path) -> None:
    report = EvalRunner(settings=eval_settings(tmp_path)).run("routing")

    assert report.failed == 0
    assert report.by_category["routing"].passed == report.by_category["routing"].total
    assert any(result.fallback_from == "codex" for result in report.results)


def test_tool_eval_passes_with_grounded_model_calls(tmp_path: Path) -> None:
    report = EvalRunner(settings=eval_settings(tmp_path)).run("tools")

    assert report.failed == 0
    grounded = [result for result in report.results if result.tool_name]
    assert grounded
    assert all(result.model_called for result in grounded)
    assert {result.tool_name for result in grounded} == {
        "time",
        "unsupported_live_data",
        "stock_price",
        "web_search",
    }
    pass_through = [
        result
        for result in report.results
        if result.expected_capability in {"weather", "latest_info", "stock_price"}
        and not result.tool_name
    ]
    assert pass_through
    assert all(result.model_called for result in pass_through)


def test_session_eval_passes_and_checks_context_recall(tmp_path: Path) -> None:
    report = EvalRunner(settings=eval_settings(tmp_path)).run("session")

    assert report.failed == 0
    assert all(result.notes["same_session"] for result in report.results)
    assert all(result.notes["context_recall_observed"] for result in report.results)


def test_eval_result_records_metrics_metadata(tmp_path: Path) -> None:
    report = EvalRunner(settings=eval_settings(tmp_path)).run("routing", limit=1)
    result = report.results[0]

    assert result.metrics_recorded
    assert result.request_id is not None
    assert result.raw_answer_preview


def test_scorer_reports_backend_mismatch() -> None:
    case = EvalCase(
        case_id="case",
        category="routing",
        name="Mismatch",
        prompt="Debug this repo",
        expected_backend="codex",
    )
    result = EvalResult(
        case_id="case",
        category="routing",
        name="Mismatch",
        passed=False,
        expected_backend="codex",
        selected_backend="ollama",
        should_call_model=True,
        model_called=True,
        success=True,
        metrics_recorded=True,
    )

    scored = score_case(case, result)

    assert not scored.passed
    assert "expected backend codex" in (scored.failure_reason or "")


def test_eval_report_json_contains_summary() -> None:
    result = EvalResult(
        case_id="case",
        category="routing",
        name="Case",
        passed=True,
        success=True,
        metrics_recorded=True,
    )

    payload = json.loads(report_to_json(build_report("routing", [result])))

    assert payload["summary"]["total"] == 1
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["skipped"] == 0
    assert payload["summary"]["not_verified"] == 0
    assert payload["mode"] == "mock"
    assert payload["failures"] == []


def test_eval_report_counts_statuses() -> None:
    results = [
        EvalResult(case_id="pass", category="routing", name="Pass", passed=True),
        EvalResult(
            case_id="fail",
            category="routing",
            name="Fail",
            passed=False,
            status=EvalStatus.FAIL,
            failure_reason="boom",
        ),
        EvalResult(
            case_id="skip",
            category="tools",
            name="Skip",
            passed=False,
            status=EvalStatus.SKIPPED,
        ),
        EvalResult(
            case_id="missing",
            category="real",
            name="Missing",
            passed=False,
            status=EvalStatus.NOT_VERIFIED,
        ),
    ]

    report = build_report(
        "mixed",
        results,
        mode="real",
        backend_availability={"codex": False},
    )
    payload = json.loads(report_to_json(report))

    assert report.passed == 1
    assert report.failed == 1
    assert report.skipped == 1
    assert report.not_verified == 1
    assert payload["mode"] == "real"
    assert payload["backend_availability"] == {"codex": False}
    assert payload["summary"]["not_verified"] == 1


def unavailable_registry_factory(container, cwd):  # noqa: ANN001
    adapters = {
        "ollama": MockAgentAdapter("ollama", available=False),
        "codex": MockAgentAdapter("codex", available=False),
        "claude-code": MockAgentAdapter("claude-code", available=False),
    }
    registry_adapters: dict[str, AgentAdapter] = dict(adapters)
    return BackendRegistry(registry_adapters)


class TimeoutAdapter(AgentAdapter):
    def __init__(self, name: str) -> None:
        self.name = name
        self.cost_type = BackendCostType.SUBSCRIPTION

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            selected_model=f"{self.name}/timeout-test",
            latency_ms=request.timeout_s * 1000,
            success=False,
            error_message=(
                f"{self.name} timed out after {request.timeout_s}s. "
                "Runtime context: secret prompt body"
            ),
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


def timeout_registry_factory(container, cwd):  # noqa: ANN001
    adapters: dict[str, AgentAdapter] = {
        "ollama": MockAgentAdapter("ollama", available=False),
        "codex": TimeoutAdapter("codex"),
        "claude-code": MockAgentAdapter("claude-code", available=False),
    }
    return BackendRegistry(adapters)


def test_real_smoke_marks_unavailable_backends_not_verified(tmp_path: Path) -> None:
    report = RealSmokeRunner(
        settings=eval_settings(tmp_path),
        registry_factory=unavailable_registry_factory,
        timeout_s=1,
    ).run()

    assert report.mode == "real"
    assert report.failed == 0
    assert report.not_verified > 0
    assert report.passed == 0
    assert report.backend_availability == {
        "claude-code": False,
        "codex": False,
        "ollama": False,
    }
    not_verified = [
        result for result in report.results if result.status == EvalStatus.NOT_VERIFIED
    ]
    assert all(
        "required backend unavailable" in (result.failure_reason or "")
        for result in not_verified
    )


def test_real_smoke_json_includes_mode_availability_and_not_verified(
    tmp_path: Path,
) -> None:
    report = RealSmokeRunner(
        settings=eval_settings(tmp_path),
        registry_factory=unavailable_registry_factory,
        timeout_s=1,
    ).run(limit=5)
    payload = json.loads(report_to_json(report))

    assert payload["mode"] == "real"
    assert payload["backend_availability"]["codex"] is False
    assert payload["summary"]["not_verified"] >= 1


def test_eval_real_smoke_command_does_not_fail_when_backends_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "switchboard.cli.build_real_smoke_runner",
        lambda *, timeout_s=90, case_timeouts=None: RealSmokeRunner(
            settings=eval_settings(tmp_path),
            registry_factory=unavailable_registry_factory,
            timeout_s=timeout_s,
            case_timeouts=case_timeouts,
        ),
    )

    eval_real_smoke_command(
        argparse.Namespace(
            json=False,
            output=None,
            limit=5,
            timeout=1,
            case_timeout=[],
            fast=False,
            tag=None,
        )
    )

    output = capsys.readouterr().out
    assert "Mode: real" in output
    assert "Not verified:" in output


def test_real_provider_eval_marks_missing_providers_not_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.delenv("SWITCHBOARD_WEB_PROVIDER", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SWITCHBOARD_FINANCE_PROVIDER", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)

    report = RealProviderRunner(
        settings=eval_settings(
            tmp_path,
            personal_config_path=live_provider_config(tmp_path),
        )
    ).run()

    assert report.failed == 0
    assert report.not_verified == report.total
    assert all(result.status == EvalStatus.NOT_VERIFIED for result in report.results)


def test_real_provider_eval_uses_personal_finance_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.delenv("SWITCHBOARD_FINANCE_PROVIDER", raising=False)
    calls: list[str] = []

    class FakeFinanceProvider:
        def is_configured(self) -> bool:
            return True

        def get_quote(self, symbol: str) -> StockQuote:
            return StockQuote(
                symbol=symbol.upper(),
                price=123.45,
                currency="USD",
                source="Fake Finance",
            )

    def fake_finance_provider_by_name(name: str) -> FakeFinanceProvider:
        calls.append(name)
        return FakeFinanceProvider()

    monkeypatch.setattr(
        "switchboard.evals.real_providers.finance_provider_by_name",
        fake_finance_provider_by_name,
    )

    runner = RealProviderRunner(
        settings=eval_settings(
            tmp_path,
            personal_config_path=live_provider_config(tmp_path, finance_provider="yahoo"),
        )
    )
    result = runner._finance_provider_status("NOW")

    assert result.status == EvalStatus.PASS
    assert calls == ["yahoo"]


def test_real_provider_eval_explicit_none_disables_env_finance_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "yahoo")
    calls: list[str] = []

    def fake_finance_provider_by_name(name: str):  # noqa: ANN001
        calls.append(name)
        return UnconfiguredFinanceProvider()

    monkeypatch.setattr(
        "switchboard.evals.real_providers.finance_provider_by_name",
        fake_finance_provider_by_name,
    )

    runner = RealProviderRunner(
        settings=eval_settings(
            tmp_path,
            personal_config_path=live_provider_config(tmp_path, finance_provider="none"),
        )
    )
    result = runner._finance_provider_status("NOW")

    assert result.status == EvalStatus.NOT_VERIFIED
    assert calls == ["none"]


def test_real_provider_eval_empty_finance_config_uses_env_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setenv("SWITCHBOARD_FINANCE_PROVIDER", "yahoo")
    calls: list[str] = []

    def fake_get_quote(self, symbol: str) -> StockQuote:  # noqa: ANN001
        calls.append(symbol)
        return StockQuote(
            symbol=symbol.upper(),
            price=123.45,
            currency="USD",
            source="Fake Finance",
        )

    monkeypatch.setattr(
        "switchboard.app.services.finance_providers.YahooFinanceProvider.get_quote",
        fake_get_quote,
    )

    runner = RealProviderRunner(
        settings=eval_settings(
            tmp_path,
            personal_config_path=live_provider_config(tmp_path, finance_provider=""),
        )
    )
    result = runner._finance_provider_status("NOW")

    assert result.status == EvalStatus.PASS
    assert calls == ["NOW"]


def test_eval_real_providers_command_does_not_fail_when_providers_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # noqa: ANN001
    monkeypatch.delenv("SWITCHBOARD_WEB_PROVIDER", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("SWITCHBOARD_FINANCE_PROVIDER", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(
        "switchboard.cli.RealProviderRunner",
        lambda *, timeout_s=120: RealProviderRunner(
            settings=eval_settings(
                tmp_path,
                personal_config_path=live_provider_config(tmp_path),
            ),
            timeout_s=timeout_s,
        ),
    )

    eval_real_providers_command(argparse.Namespace(json=False, output=None, timeout=1))

    output = capsys.readouterr().out
    assert "Suite: real-providers" in output
    assert "Not verified:" in output


def test_real_smoke_timeout_status_records_diagnostics(tmp_path: Path) -> None:
    report = RealSmokeRunner(
        settings=eval_settings(tmp_path),
        registry_factory=timeout_registry_factory,
        timeout_s=2,
    ).run(limit=1)
    result = report.results[0]

    assert result.status == EvalStatus.TIMEOUT
    assert report.timed_out == 1
    assert result.prompt
    assert result.selected_backend == "codex"
    assert result.timeout_seconds == 2
    assert result.elapsed_seconds is not None
    assert result.error_type == "timeout"
    assert result.process_started is True
    assert result.process_exited is False
    assert "secret prompt body" not in (result.sanitized_error or "")
    assert result.notes["route_selection_status"] == "PASS"
    assert result.notes["route_selection_passed"] is True


def test_real_smoke_fast_tag_filters_cases(tmp_path: Path) -> None:
    report = RealSmokeRunner(
        settings=eval_settings(tmp_path),
        registry_factory=unavailable_registry_factory,
        timeout_s=1,
    ).run(tags={"fast"})

    assert report.total < len(cases_for_suite("real-smoke"))
    assert report.total > 0


def test_eval_real_smoke_command_accepts_timeout_fast_and_case_timeout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}

    def fake_builder(*, timeout_s=90, case_timeouts=None):  # noqa: ANN001
        captured["timeout_s"] = timeout_s
        captured["case_timeouts"] = case_timeouts
        return RealSmokeRunner(
            settings=eval_settings(tmp_path),
            registry_factory=unavailable_registry_factory,
            timeout_s=timeout_s,
            case_timeouts=case_timeouts,
        )

    monkeypatch.setattr("switchboard.cli.build_real_smoke_runner", fake_builder)

    eval_real_smoke_command(
        argparse.Namespace(
            json=False,
            output=None,
            limit=None,
            timeout=7,
            case_timeout=["real_auto_coding=11"],
            fast=True,
            tag=None,
        )
    )

    assert captured["timeout_s"] == 7
    assert captured["case_timeouts"] == {"real_auto_coding": 11}
    assert "Mode: real" in capsys.readouterr().out


def test_eval_command_dry_run_lists_cases(capsys) -> None:  # noqa: ANN001
    eval_command(
        argparse.Namespace(
            suite="routing",
            dry_run=True,
            json=False,
            output=None,
            backend="auto",
            mock=True,
            limit=1,
        )
    )

    output = capsys.readouterr().out
    assert "Switchboard Eval Cases" in output
    assert "route_coding_codex" in output


def test_eval_command_writes_json_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:  # noqa: ANN001
    output_path = tmp_path / "eval_results.json"
    monkeypatch.setattr(
        "switchboard.cli.build_eval_runner",
        lambda *, mock=True: EvalRunner(settings=eval_settings(tmp_path), mock=mock),
    )

    eval_command(
        argparse.Namespace(
            suite="tools",
            dry_run=False,
            json=False,
            output=str(output_path),
            backend="auto",
            mock=True,
            limit=None,
        )
    )

    output = capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "Wrote JSON report" in output
    assert payload["summary"]["failed"] == 0
