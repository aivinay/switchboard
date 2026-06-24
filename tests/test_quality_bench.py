"""Structural tests for the quality benchmark and its dataset."""

from __future__ import annotations

from pathlib import Path

from switchboard.app.core.config import Settings
from switchboard.evals.quality_bench import (
    DEFAULT_CONDITIONS,
    BenchCondition,
    MockJudge,
    QualityBenchRunner,
    report_to_text,
)
from switchboard.evals.quality_dataset import cases_by_category, quality_cases

ROOT = Path(__file__).resolve().parents[1]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'bench.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(ROOT / "config" / "personal.yaml"),
    )


def test_dataset_has_100_cases_with_expected_categories() -> None:
    cases = quality_cases()
    assert len(cases) == 100
    grouped = cases_by_category()
    assert len(grouped["coding"]) == 25
    assert len(grouped["reasoning"]) == 25
    assert len(grouped["summarization"]) == 25
    assert len(grouped["private"]) == 15
    assert len(grouped["grounding"]) == 10
    assert len({case.case_id for case in cases}) == 100


def test_private_cases_are_marked_local_only() -> None:
    for case in cases_by_category()["private"]:
        assert case.expected_local_only
    for category in ("coding", "reasoning", "summarization", "grounding"):
        for case in cases_by_category()[category]:
            assert not case.expected_local_only


def test_mock_judge_is_deterministic() -> None:
    case = quality_cases()[0]
    judge = MockJudge()
    first = judge.score(case, "an answer mentioning " + " ".join(case.keywords))
    second = judge.score(case, "an answer mentioning " + " ".join(case.keywords))
    assert first.score == second.score == 5.0
    assert judge.score(case, "").score == 1.0


def test_mock_bench_run_produces_condition_summaries(tmp_path: Path) -> None:
    runner = QualityBenchRunner(
        settings=make_settings(tmp_path),
        mock=True,
        conditions=(
            BenchCondition(name="always_premium", forced_backend="claude-code"),
            BenchCondition(name="rules"),
        ),
    )
    report = runner.run(limit=6)

    assert report["mode"] == "mock"
    assert report["case_count"] == 6
    conditions = report["conditions"]
    assert [c["condition"] for c in conditions] == ["always_premium", "rules"]
    premium = conditions[0]
    assert premium["cases"] == 6
    assert premium["premium_rate"] == 1.0  # everything forced to claude-code
    # The stratified sample includes one private-category case. Since the
    # round-4 keyword-floor broadening ("my symptoms" et al.), private mode
    # blocks it on a forced subscription backend: a failure, never a leak.
    blocked = [
        record
        for record in report["records"]
        if record["condition"] == "always_premium" and not record["success"]
    ]
    assert all(record["category"] == "private" for record in blocked)
    assert all(not record["privacy_violation"] for record in blocked)
    assert premium["success_rate"] == round((6 - len(blocked)) / 6, 4)
    assert len(report["records"]) == 12
    assert report_to_text(report)  # renders without crashing


def test_private_case_privacy_accounting_is_consistent(tmp_path: Path) -> None:
    runner = QualityBenchRunner(
        settings=make_settings(tmp_path),
        mock=True,
        conditions=(
            BenchCondition(name="rules"),
            BenchCondition(name="always_premium", forced_backend="claude-code"),
        ),
    )
    report = runner.run(categories=("private",))

    rules, premium = report["conditions"]
    assert rules["cases"] == premium["cases"] == 15
    # A violation is a private prompt actually ANSWERED by a non-local
    # backend. Blocked requests are failures, not leaks.
    for record in report["records"]:
        if record["privacy_violation"]:
            assert record["success"] and record["backend"] not in {"ollama"}
    # Switchboard routing must leak strictly less than the always-premium
    # baseline (the paper's central privacy comparison).
    assert rules["privacy_violations"] < premium["privacy_violations"]


def test_always_premium_baseline_violates_privacy(tmp_path: Path) -> None:
    runner = QualityBenchRunner(
        settings=make_settings(tmp_path),
        mock=True,
        conditions=(BenchCondition(name="always_premium", forced_backend="claude-code"),),
    )
    report = runner.run(categories=("private",))

    premium = report["conditions"][0]
    # Forcing the premium backend leaks every private prompt that private
    # mode's sensitivity classifier fails to block.
    answered = [r for r in report["records"] if r["success"] and r["answer_preview"]]
    assert premium["privacy_violations"] == len(answered)
    assert premium["privacy_violations"] >= 1


def test_limit_samples_across_all_categories(tmp_path: Path) -> None:
    runner = QualityBenchRunner(
        settings=make_settings(tmp_path),
        mock=True,
        conditions=(BenchCondition(name="rules"),),
    )
    report = runner.run(limit=10)

    assert report["case_count"] == 10
    categories = {record["category"] for record in report["records"]}
    assert categories == {"coding", "reasoning", "summarization", "private", "grounding"}


def test_default_conditions_cover_baselines_and_ablations() -> None:
    names = {condition.name for condition in DEFAULT_CONDITIONS}
    assert {"always_premium", "always_local", "rules", "hybrid", "llm", "hybrid_full"} <= names
