from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def unix_timestamp() -> int:
    return int(utc_now().timestamp())
