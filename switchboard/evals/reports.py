from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from switchboard.evals.types import (
    EvalCategorySummary,
    EvalReport,
    EvalResult,
    EvalStatus,
)


def build_report(
    suite: str,
    results: list[EvalResult],
    *,
    mode: str = "mock",
    backend_availability: dict[str, bool] | None = None,
) -> EvalReport:
    total = len(results)
    passed = sum(1 for result in results if result.status == EvalStatus.PASS)
    failed = sum(1 for result in results if result.status == EvalStatus.FAIL)
    timed_out = sum(1 for result in results if result.status == EvalStatus.TIMEOUT)
    skipped = sum(1 for result in results if result.status == EvalStatus.SKIPPED)
    not_verified = sum(1 for result in results if result.status == EvalStatus.NOT_VERIFIED)
    latency_by_backend: dict[str, list[float]] = {}
    timeouts_by_backend_counter: Counter[str] = Counter()
    for result in results:
        backend = result.selected_backend or result.expected_backend or "unknown"
        if result.elapsed_seconds is not None:
            latency_by_backend.setdefault(backend, []).append(result.elapsed_seconds)
        if result.status == EvalStatus.TIMEOUT:
            timeouts_by_backend_counter[backend] += 1
    average_latency = {
        backend: round(sum(values) / len(values), 3)
        for backend, values in sorted(latency_by_backend.items())
        if values
    }
    by_category: dict[str, EvalCategorySummary] = {}
    counts = Counter(result.category for result in results)
    status_counts = Counter((result.category, result.status) for result in results)
    for category in sorted(counts):
        category_total = counts[category]
        category_passed = status_counts[(category, EvalStatus.PASS)]
        by_category[category] = EvalCategorySummary(
            total=category_total,
            passed=category_passed,
            failed=status_counts[(category, EvalStatus.FAIL)],
            timed_out=status_counts[(category, EvalStatus.TIMEOUT)],
            skipped=status_counts[(category, EvalStatus.SKIPPED)],
            not_verified=status_counts[(category, EvalStatus.NOT_VERIFIED)],
        )
    return EvalReport(
        suite=suite,
        mode=mode,
        total=total,
        passed=passed,
        failed=failed,
        timed_out=timed_out,
        skipped=skipped,
        not_verified=not_verified,
        by_category=by_category,
        failures=[
            result
            for result in results
            if result.status in {EvalStatus.FAIL, EvalStatus.TIMEOUT}
        ],
        results=results,
        backend_availability=backend_availability or {},
        average_latency_seconds_by_backend=average_latency,
        timeouts_by_backend=dict(sorted(timeouts_by_backend_counter.items())),
    )


def report_to_text(report: EvalReport) -> str:
    lines = [
        "Switchboard Eval Report",
        "-----------------------",
        f"Suite: {report.suite}",
        f"Mode: {report.mode}",
        f"Cases: {report.total}",
        f"Passed: {report.passed}",
        f"Failed: {report.failed}",
        f"Timed out: {report.timed_out}",
        f"Skipped: {report.skipped}",
        f"Not verified: {report.not_verified}",
        f"Pass rate excluding skipped/not verified: {report.success_rate:.0%}",
    ]
    if report.backend_availability:
        lines.extend(["", "Backend availability:"])
        for backend, available in sorted(report.backend_availability.items()):
            status = "available" if available else "unavailable"
            lines.append(f"- {backend}: {status}")
    if report.average_latency_seconds_by_backend:
        lines.extend(["", "Average latency by backend:"])
        for backend, latency in report.average_latency_seconds_by_backend.items():
            lines.append(f"- {backend}: {latency:.3f}s")
    if report.timeouts_by_backend:
        lines.extend(["", "Timeouts by backend:"])
        for backend, count in report.timeouts_by_backend.items():
            lines.append(f"- {backend}: {count}")
    lines.extend(["", "By category:"])
    if report.by_category:
        for category, summary in report.by_category.items():
            lines.append(
                f"- {category}: {summary.passed}/{summary.total} passed, "
                f"{summary.failed} failed, {summary.timed_out} timed out, "
                f"{summary.skipped} skipped, {summary.not_verified} not verified"
            )
    else:
        lines.append("- none")
    lines.extend(["", "Failures and timeouts:"])
    if report.failures:
        for failure in report.failures:
            selected = failure.selected_backend or "-"
            elapsed = (
                f"{failure.elapsed_seconds:.3f}s"
                if failure.elapsed_seconds is not None
                else "-"
            )
            timeout = f"{failure.timeout_seconds}s" if failure.timeout_seconds else "-"
            reason = failure.sanitized_error or failure.failure_reason or "failed without reason"
            lines.append(
                f"- {failure.case_id} {selected} elapsed={elapsed} "
                f"timeout={timeout} status={failure.status}: {reason}"
            )
    else:
        lines.append("- none")
    not_verified = [
        result for result in report.results if result.status == EvalStatus.NOT_VERIFIED
    ]
    if not_verified:
        lines.extend(["", "Not verified:"])
        for result in not_verified:
            lines.append(f"- {result.case_id}: {result.failure_reason or 'backend unavailable'}")
    return "\n".join(lines)


def report_to_json(report: EvalReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


def write_report(path: Path, report: EvalReport) -> None:
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
