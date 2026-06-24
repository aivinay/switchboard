"""Tests for the strengthened capability detector and deterministic tools."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.core.config import Settings
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.models.capabilities import Capability, RuntimeContext
from switchboard.app.services.capabilities import CapabilityDetector
from switchboard.app.services.container import build_container
from switchboard.app.services.deterministic_tools import (
    CalculatorTool,
    UnitConversionTool,
)
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.tools import TimeTool
from switchboard.app.storage.db import create_db_engine, init_db

ROOT = Path(__file__).resolve().parents[1]

detector = CapabilityDetector()


def runtime_context() -> RuntimeContext:
    utc_now = datetime(2026, 6, 11, 16, 30, tzinfo=UTC)
    return RuntimeContext(
        utc_datetime=utc_now,
        local_datetime=utc_now,
        local_timezone="Asia/Kolkata",
        current_date="2026-06-11",
        utc_iso=utc_now.isoformat(),
        local_iso=utc_now.isoformat(),
        human_utc_time="4:30 PM UTC",
        human_local_time="10:00 PM IST",
    )


# ---------------------------------------------------------------------------
# Capability detection breadth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "capability"),
    [
        ("is it raining in mumbai?", Capability.WEATHER),
        ("is it gonna rain in seattle tomorrow?", Capability.WEATHER),
        ("will there be rain tomorrow?", Capability.WEATHER),
        ("is it going to snow tonight", Capability.WEATHER),
        ("any chance of showers this weekend?", Capability.WEATHER),
        ("how hot is it outside", Capability.WEATHER),
        ("do i need an umbrella today", Capability.WEATHER),
        ("what's the air quality in delhi", Capability.WEATHER),
        ("when is sunset today", Capability.WEATHER),
        ("how is tesla stock doing", Capability.STOCK_PRICE),
        ("shares of infosys", Capability.STOCK_PRICE),
        ("stock quote for reliance", Capability.STOCK_PRICE),
        ("what is the exchange rate for usd to inr", Capability.LATEST_INFO),
        ("price of bitcoin", Capability.LATEST_INFO),
        ("gold price today", Capability.LATEST_INFO),
        ("who won the cricket match", Capability.LATEST_INFO),
        ("election results in maharashtra", Capability.LATEST_INFO),
        # Corporate events are live facts (dogfood: fabricated SpaceX IPO).
        ("did spacex go public", Capability.LATEST_INFO),
        ("has openai been acquired", Capability.LATEST_INFO),
        ("when did arm get listed on the nasdaq", Capability.LATEST_INFO),
        ("is wework bankrupt", Capability.LATEST_INFO),
        ("google this for me: best laptops", Capability.WEB_SEARCH),
        ("can you search the internet", Capability.WEB_SEARCH),
        ("what year is it", Capability.CURRENT_DATE),
        ("what day is tomorrow", Capability.CURRENT_DATE),
        ("yesterday's date", Capability.CURRENT_DATE),
        ("what date is 45 days from today", Capability.CURRENT_DATE),
        ("what will the date be in 100 days", Capability.CURRENT_DATE),
        ("15 days from today, what will the date be", Capability.CURRENT_DATE),
        ("tell me the date 5 days from now", Capability.CURRENT_DATE),
        ("what's the local time in tokyo", Capability.CURRENT_TIME),
    ],
)
def test_detector_breadth(prompt: str, capability: Capability) -> None:
    assert detector.detect(prompt).has(capability), prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "what is 234 * 78?",
        "calculate 15% of 240",
        "square root of 144",
        "2+2",
        "how much is 1200 / 30",
    ],
)
def test_calculation_detected(prompt: str) -> None:
    assert detector.detect(prompt).has(Capability.CALCULATION), prompt


@pytest.mark.parametrize(
    "prompt",
    [
        "convert 10 km to miles",
        "100 fahrenheit in celsius",
        "5 kg to lbs",
        "2 liters in gallons",
    ],
)
def test_unit_conversion_detected(prompt: str) -> None:
    assert detector.detect(prompt).has(Capability.UNIT_CONVERSION), prompt


def test_arithmetic_inside_code_prompt_does_not_hijack_primary() -> None:
    detection = detector.detect("fix this python bug: result = compute(2+2) raises error")
    assert detection.has(Capability.CODING)


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("what is 234 * 78", "18,252"),
        ("calculate 15% of 240", "36"),
        ("square root of 144", "12"),
        ("(3 + 5) / 2", "4"),
        ("2^10", "1,024"),
        ("what is 1,000 + 2,000", "3,000"),
    ],
)
def test_calculator_results(prompt: str, expected: str) -> None:
    result = CalculatorTool().answer(prompt)
    assert result.success, result.error
    assert f"= {expected}." in result.answer


def test_calculator_division_by_zero_fails_gracefully() -> None:
    result = CalculatorTool().answer("what is 5 / 0")
    assert not result.success
    assert "zero" in (result.error or "").lower()


def test_calculator_rejects_huge_exponents() -> None:
    result = CalculatorTool().answer("what is 9999^9999")
    assert not result.success


# ---------------------------------------------------------------------------
# Unit conversion tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "fragment"),
    [
        ("convert 10 km to miles", "6.213712 miles"),
        ("100 fahrenheit in celsius", "37.7778°C"),
        ("0 celsius to fahrenheit", "32°F"),
        ("5 kg to lbs", "11.02311"),
        ("12 inches in cm", "30.48 cm"),
    ],
)
def test_unit_conversions(prompt: str, fragment: str) -> None:
    result = UnitConversionTool().answer(prompt)
    assert result.success, result.error
    assert fragment in result.answer, result.answer


def test_unit_conversion_rejects_cross_kind() -> None:
    result = UnitConversionTool().answer("convert 5 kg to miles")
    assert not result.success
    assert "Cannot convert" in (result.error or "")


# ---------------------------------------------------------------------------
# TimeTool: timezones and date arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "label"),
    [
        ("time in tokyo", "Tokyo"),
        ("what time is it in san francisco", "San Francisco"),
        ("time in bengaluru", "Bengaluru"),
        ("current time in dubai", "Dubai"),
        ("time in sydney", "Sydney"),
        ("what's the time in berlin", "Berlin"),
    ],
)
def test_timetool_city_coverage(prompt: str, label: str) -> None:
    result = TimeTool().answer(
        prompt=prompt, capability=Capability.CURRENT_TIME, context=runtime_context()
    )
    assert f"in {label}" in result.answer


def test_timetool_short_alias_does_not_match_inside_words() -> None:
    # "interest" contains "est"; must not resolve to US Eastern.
    result = TimeTool().answer(
        prompt="what time is it? asking out of interest",
        capability=Capability.CURRENT_TIME,
        context=runtime_context(),
    )
    assert "US Eastern" not in result.answer


def test_timetool_date_includes_weekday() -> None:
    result = TimeTool().answer(
        prompt="what's the date today",
        capability=Capability.CURRENT_DATE,
        context=runtime_context(),
    )
    assert "Today is Thursday, June 11, 2026" in result.answer


@pytest.mark.parametrize(
    ("prompt", "fragment"),
    [
        ("what day is tomorrow", "Tomorrow is Friday, June 12, 2026"),
        ("what day was yesterday", "Yesterday was Wednesday, June 10, 2026"),
        ("what date is the day after tomorrow", "The day after tomorrow is Saturday, June 13"),
        ("what date is 45 days from today", "45 days from today is Sunday, July 26, 2026"),
        ("what will the date be in 7 days", "7 days from today is Thursday, June 18, 2026"),
        ("tell me the date 1 day from now", "1 day from today is Friday, June 12, 2026"),
    ],
)
def test_timetool_date_arithmetic(prompt: str, fragment: str) -> None:
    result = TimeTool().answer(
        prompt=prompt, capability=Capability.CURRENT_DATE, context=runtime_context()
    )
    assert fragment in result.answer, result.answer


# ---------------------------------------------------------------------------
# End to end: grounded answers route local
# ---------------------------------------------------------------------------


class FakeAdapter(AgentAdapter):
    def __init__(self, name: str, *, cost_type: BackendCostType = BackendCostType.LOCAL) -> None:
        self.name = name
        self.cost_type = cost_type
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return True

    def availability(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, cost_type=self.cost_type)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        self.prompts.append(request.prompt)
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=f"{self.name} answered",
            latency_ms=5,
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


def make_service(tmp_path: Path) -> tuple[SwitchboardCoreService, dict[str, FakeAdapter]]:
    adapters = {
        "ollama": FakeAdapter("ollama"),
        "codex": FakeAdapter("codex", cost_type=BackendCostType.SUBSCRIPTION),
        "claude-code": FakeAdapter("claude-code", cost_type=BackendCostType.SUBSCRIPTION),
    }
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'strength.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)
    container.personal_config.preferences.claude_code_web_search = False
    service = SwitchboardCoreService(
        registry=BackendRegistry(dict(adapters)),
        metrics=container.backend_metrics_repository,
        container=container,
    )
    return service, adapters


@pytest.mark.parametrize(
    ("prompt", "fact_fragment"),
    [
        ("what is 234 * 78?", "= 18,252"),
        ("convert 10 km to miles", "6.213712 miles"),
        ("what's the date today", "Today is"),
    ],
)
def test_grounded_prompts_route_local_with_trusted_fact(
    tmp_path: Path, prompt: str, fact_fragment: str
) -> None:
    service, adapters = make_service(tmp_path)

    response = service.ask(prompt, new_session=True)

    assert response.success
    assert response.backend == "ollama"
    assert "deterministic tool grounded" in response.routing_reason.lower()
    assert fact_fragment in adapters["ollama"].prompts[-1]
    assert adapters["claude-code"].prompts == []
    assert adapters["codex"].prompts == []


def test_grounded_date_with_coding_request_still_goes_to_codex(tmp_path: Path) -> None:
    service, adapters = make_service(tmp_path)

    response = service.ask(
        "write a python script that prints today's date", new_session=True
    )

    assert response.backend == "codex"
    # The trusted date fact still reaches Codex so the script comment is right.
    assert "Today is" in adapters["codex"].prompts[-1]


def test_failed_calculator_does_not_get_live_data_honesty_fact(tmp_path: Path) -> None:
    service, adapters = make_service(tmp_path)

    response = service.ask("what is 5 / 0", new_session=True)

    assert response.success  # passes through to a model
    prompt = (adapters["ollama"].prompts + adapters["claude-code"].prompts)[-1]
    assert "cannot access live data" not in prompt
