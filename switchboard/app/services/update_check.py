from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from switchboard.app.core.config import user_config_dir

PACKAGE_NAME = "switchboard-local"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
UPDATE_CHECK_FILENAME = "update-check.json"


@dataclass(frozen=True)
class VersionStatus:
    installed: str
    latest: str | None
    update_available: bool


def update_check_cache_path() -> Path:
    return user_config_dir() / UPDATE_CHECK_FILENAME


def read_update_cache(cache_path: Path | None = None) -> dict[str, Any]:
    path = cache_path or update_check_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
) -> VersionStatus:
    cache = read_update_cache(cache_path)
    latest = cache.get("latest")
    latest_version = latest if isinstance(latest, str) and latest else None
    return VersionStatus(
        installed=installed_version,
        latest=latest_version,
        update_available=is_newer_version(latest_version, installed_version),
    )
