from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from switchboard.app.models.capabilities import RuntimeContext

Clock = Callable[[], datetime]


class RuntimeContextProvider:
    def __init__(
        self,
        *,
        local_timezone: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.local_timezone = local_timezone or self._system_timezone() or "America/New_York"
        self.clock = clock or (lambda: datetime.now(UTC))

    def current(self) -> RuntimeContext:
        utc_now = self._aware_utc_now()
        local_zone = self._zone(self.local_timezone)
        local_now = utc_now.astimezone(local_zone)
        return RuntimeContext(
            utc_datetime=utc_now,
            local_datetime=local_now,
            local_timezone=self.local_timezone,
            current_date=self._format_date(local_now),
            utc_iso=utc_now.isoformat(),
            local_iso=local_now.isoformat(),
            human_utc_time=self._human_time(utc_now),
            human_local_time=self._human_time(local_now),
        )

    def _aware_utc_now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=UTC)
        return now.astimezone(UTC)

    def _system_timezone(self) -> str | None:
        timezone = os.environ.get("TZ")
        if timezone and self._is_valid_timezone(timezone):
            return timezone
        return None

    def _zone(self, timezone: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            self.local_timezone = "America/New_York"
            return ZoneInfo(self.local_timezone)

    def _is_valid_timezone(self, timezone: str) -> bool:
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError:
            return False
        return True

    def _format_date(self, value: datetime) -> str:
        return f"{value:%B} {value.day}, {value:%Y}"

    def _human_time(self, value: datetime) -> str:
        hour = value.strftime("%I").lstrip("0") or "0"
        abbreviation = value.tzname() or "UTC"
        return f"{hour}:{value:%M} {value:%p} {abbreviation} on {self._format_date(value)}"
