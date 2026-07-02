from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from switchboard.app.models.personal import PersonalQuotaConfig
from switchboard.app.storage.repositories import BackendMetricsRepository


@dataclass(frozen=True)
class QuotaWindowSpec:
    backend: str
    label: str
    budget_field: str
    window: timedelta
    window_label: str


@dataclass(frozen=True)
class QuotaWindowStatus:
    backend: str
    label: str
    window: str
    window_seconds: int
    used: int
    budget: int | None
    remaining: int | None
    constrained: bool
    since: datetime
    until: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "label": self.label,
            "window": self.window,
            "window_seconds": self.window_seconds,
            "used": self.used,
            "budget": self.budget,
            "remaining": self.remaining,
            "constrained": self.constrained,
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
        }


PREMIUM_QUOTA_WINDOWS: tuple[QuotaWindowSpec, ...] = (
    QuotaWindowSpec(
        backend="codex",
        label="Codex",
        budget_field="codex_calls_per_5h",
        window=timedelta(hours=5),
        window_label="5h",
    ),
    QuotaWindowSpec(
        backend="claude-code",
        label="Claude",
        budget_field="claude_calls_per_week",
        window=timedelta(days=7),
        window_label="7d",
    ),
)

PREMIUM_BACKENDS = frozenset(spec.backend for spec in PREMIUM_QUOTA_WINDOWS)


class QuotaLedgerService:
    """Estimate-only premium quota ledger derived from local backend metrics."""

    def __init__(
        self,
        metrics: BackendMetricsRepository,
        quota: PersonalQuotaConfig,
        *,
        now_factory: Callable[[], datetime] | None = None,
    ) -> None:
        self.metrics = metrics
        self.quota = quota
        self.now_factory = now_factory or (lambda: datetime.now(UTC))

    def enabled(self) -> bool:
        return any(self._budget_for_spec(spec) is not None for spec in PREMIUM_QUOTA_WINDOWS)

    def snapshot(self, *, now: datetime | None = None) -> dict[str, object]:
        current = now or self.now_factory()
        windows = {spec.backend: self.status_for_spec(spec, now=current).to_dict()
            for spec in PREMIUM_QUOTA_WINDOWS}
        return {
            "enabled": any(window["budget"] is not None for window in windows.values()),
            "source": "local_backend_metrics",
            "policy": "user_declared_soft_budget",
            "windows": windows,
        }

    def status_for_backend(
        self,
        backend: str,
        *,
        now: datetime | None = None,
    ) -> QuotaWindowStatus | None:
        spec = next(
            (
                candidate
                for candidate in PREMIUM_QUOTA_WINDOWS
                if candidate.backend == backend
            ),
            None,
        )
        if spec is None:
            return None
        return self.status_for_spec(spec, now=now or self.now_factory())

    def status_for_spec(
        self,
        spec: QuotaWindowSpec,
        *,
        now: datetime,
    ) -> QuotaWindowStatus:
        budget = self._budget_for_spec(spec)
        since = now - spec.window
        used = self.metrics.successful_call_count(
            backend=spec.backend,
            since=since,
            until=now,
        )
        remaining = None if budget is None else max(0, budget - used)
        return QuotaWindowStatus(
            backend=spec.backend,
            label=spec.label,
            window=spec.window_label,
            window_seconds=int(spec.window.total_seconds()),
            used=used,
            budget=budget,
            remaining=remaining,
            constrained=budget is not None and used >= budget,
            since=since,
            until=now,
        )

    def _budget_for_spec(self, spec: QuotaWindowSpec) -> int | None:
        budget = getattr(self.quota, spec.budget_field)
        return budget if budget is not None else None
