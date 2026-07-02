from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

from switchboard.app.core.config import user_config_dir
from switchboard.app.models.personal import PersonalConfig

PACKAGE_NAME = "switchboard-local"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
UPDATE_CHECK_FILENAME = "update-check.json"
UPDATE_CHECK_NOTICE = (
    "Switchboard checks PyPI once a day for new versions. "
    "Disable: SWITCHBOARD_UPDATE_CHECK=off."
)
UPDATE_CHECK_INTERVAL = timedelta(hours=24)
UPDATE_CHECK_TIMEOUT_S = 1.0
CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "BUILDKITE")


class HttpGet(Protocol):
    def __call__(
        self,
        url: str,
        *,
        timeout: float,
    ) -> httpx.Response: ...


@dataclass(frozen=True)
class VersionStatus:
    installed: str
    latest: str | None
    update_available: bool
    checked_at: datetime | None = None
    error: str | None = None
    skipped: bool = False


def update_check_cache_path() -> Path:
    return user_config_dir() / UPDATE_CHECK_FILENAME


def read_update_cache(cache_path: Path | None = None) -> dict[str, Any]:
    path = cache_path or update_check_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_update_cache(cache: dict[str, Any], cache_path: Path | None = None) -> None:
    path = cache_path or update_check_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        return


def _version_key(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:3])


def is_newer_version(latest: str | None, installed: str) -> bool:
    if not latest:
        return False
    latest_key = _version_key(latest)
    installed_key = _version_key(installed)
    return bool(latest_key and installed_key and latest_key > installed_key)


def cached_version_status(
    installed_version: str,
    *,
    cache_path: Path | None = None,
    skipped: bool = False,
) -> VersionStatus:
    cache = read_update_cache(cache_path)
    latest = cache.get("latest")
    latest_version = latest if isinstance(latest, str) and latest else None
    checked_at = _parse_checked_at(cache.get("checked_at"))
    error = cache.get("error")
    return VersionStatus(
        installed=installed_version,
        latest=latest_version,
        update_available=is_newer_version(latest_version, installed_version),
        checked_at=checked_at,
        error=error if isinstance(error, str) and error else None,
        skipped=skipped,
    )


def update_check_enabled(config: PersonalConfig) -> bool:
    if os.getenv("SWITCHBOARD_UPDATE_CHECK", "").strip().lower() in {"0", "off", "false", "no"}:
        return False
    if any(os.getenv(name) for name in CI_ENV_VARS):
        return False
    return config.preferences.update_check_enabled


def refresh_update_status(
    installed_version: str,
    config: PersonalConfig,
    *,
    cache_path: Path | None = None,
    now: datetime | None = None,
    http_get: HttpGet = httpx.get,
    notice: Any | None = print,
) -> VersionStatus:
    current_time = now or datetime.now(UTC)
    cache = read_update_cache(cache_path)
    if not update_check_enabled(config):
        return cached_version_status(installed_version, cache_path=cache_path, skipped=True)

    checked_at = _parse_checked_at(cache.get("checked_at"))
    if checked_at is not None and current_time - checked_at < UPDATE_CHECK_INTERVAL:
        _print_first_success_notice(cache, cache_path=cache_path, notice=notice)
        return cached_version_status(installed_version, cache_path=cache_path)

    latest = cache.get("latest")
    latest_version = latest if isinstance(latest, str) and latest else None
    notified = cache.get("notified") is True
    try:
        response = http_get(PYPI_JSON_URL, timeout=UPDATE_CHECK_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
        info = payload.get("info") if isinstance(payload, dict) else None
        fetched_latest = info.get("version") if isinstance(info, dict) else None
        if not isinstance(fetched_latest, str) or not fetched_latest:
            raise ValueError("PyPI response did not include info.version")
        latest_version = fetched_latest
        cache = {
            "checked_at": current_time.isoformat(),
            "latest": latest_version,
            "error": None,
            "notified": notified,
        }
        write_update_cache(cache, cache_path)
        _print_first_success_notice(cache, cache_path=cache_path, notice=notice)
        return VersionStatus(
            installed=installed_version,
            latest=latest_version,
            update_available=is_newer_version(latest_version, installed_version),
            checked_at=current_time,
        )
    except Exception as exc:
        cache = {
            "checked_at": current_time.isoformat(),
            "latest": latest_version,
            "error": f"{type(exc).__name__}: {exc}",
            "notified": notified,
        }
        write_update_cache(cache, cache_path)
        return VersionStatus(
            installed=installed_version,
            latest=latest_version,
            update_available=is_newer_version(latest_version, installed_version),
            checked_at=current_time,
            error=f"{type(exc).__name__}: {exc}",
        )


def _parse_checked_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _print_first_success_notice(
    cache: dict[str, Any],
    *,
    cache_path: Path | None,
    notice: Any | None,
) -> None:
    if cache.get("notified") is True or not cache.get("latest") or cache.get("error"):
        return
    if notice is not None:
        notice(UPDATE_CHECK_NOTICE)
    updated = dict(cache)
    updated["notified"] = True
    write_update_cache(updated, cache_path)
