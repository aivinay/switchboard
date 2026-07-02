from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from switchboard.app.models.personal import PersonalConfig
from switchboard.app.services.update_check import (
    CI_ENV_VARS,
    PYPI_JSON_URL,
    UPDATE_CHECK_NOTICE,
    refresh_update_status,
    update_check_cache_path,
)


class FakePyPIResponse:
    def __init__(self, version: str) -> None:
        self.version = version

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, object]:
        return {"info": {"version": self.version}}


@pytest.fixture(autouse=True)
def clear_update_check_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SWITCHBOARD_UPDATE_CHECK", raising=False)
    for name in CI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_update_check_cache_path_uses_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SWITCHBOARD_CONFIG_HOME", str(tmp_path))

    assert update_check_cache_path() == tmp_path / "switchboard" / "update-check.json"


def test_fresh_update_cache_skips_network(tmp_path: Path) -> None:
    now = datetime(2026, 7, 2, 12, tzinfo=UTC)
    cache_path = tmp_path / "update-check.json"
    cache_path.write_text(
        json.dumps(
            {
                "checked_at": (now - timedelta(hours=1)).isoformat(),
                "latest": "9.9.9",
                "notified": True,
            }
        ),
        encoding="utf-8",
    )

    status = refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=cache_path,
        now=now,
        http_get=lambda *args, **kwargs: pytest.fail("fresh cache must not call PyPI"),
        notice=None,
    )

    assert status.latest == "9.9.9"
    assert status.update_available is True


def test_stale_update_cache_fetches_with_bounded_timeout(tmp_path: Path) -> None:
    now = datetime(2026, 7, 2, 12, tzinfo=UTC)
    cache_path = tmp_path / "update-check.json"
    cache_path.write_text(
        json.dumps({"checked_at": (now - timedelta(hours=25)).isoformat(), "latest": "0.3.0"}),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_get(url: str, *, timeout: float) -> FakePyPIResponse:
        seen["url"] = url
        seen["timeout"] = timeout
        return FakePyPIResponse("0.3.1")

    status = refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=cache_path,
        now=now,
        http_get=fake_get,
        notice=None,
    )

    assert seen == {"url": PYPI_JSON_URL, "timeout": 1.0}
    assert status.latest == "0.3.1"
    assert status.update_available is True
    assert json.loads(cache_path.read_text(encoding="utf-8"))["latest"] == "0.3.1"


def test_update_check_env_opt_out_skips_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SWITCHBOARD_UPDATE_CHECK", "off")

    status = refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=tmp_path / "missing.json",
        http_get=lambda *args, **kwargs: pytest.fail("opt-out must not call PyPI"),
        notice=None,
    )

    assert status.skipped is True


def test_update_check_ci_opt_out_skips_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CI", "true")

    status = refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=tmp_path / "missing.json",
        http_get=lambda *args, **kwargs: pytest.fail("CI must not call PyPI"),
        notice=None,
    )

    assert status.skipped is True


def test_update_check_preference_opt_out_skips_network(tmp_path: Path) -> None:
    config = PersonalConfig()
    config.preferences.update_check_enabled = False

    status = refresh_update_status(
        "0.3.0",
        config,
        cache_path=tmp_path / "missing.json",
        http_get=lambda *args, **kwargs: pytest.fail("config opt-out must not call PyPI"),
        notice=None,
    )

    assert status.skipped is True


def test_successful_update_check_notice_prints_once(tmp_path: Path) -> None:
    now = datetime(2026, 7, 2, 12, tzinfo=UTC)
    cache_path = tmp_path / "update-check.json"
    notices: list[str] = []

    refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=cache_path,
        now=now,
        http_get=lambda url, *, timeout: FakePyPIResponse("0.3.0"),
        notice=notices.append,
    )
    refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=cache_path,
        now=now + timedelta(minutes=1),
        http_get=lambda *args, **kwargs: pytest.fail("fresh cache must not call PyPI"),
        notice=notices.append,
    )

    assert notices == [UPDATE_CHECK_NOTICE]
    assert json.loads(cache_path.read_text(encoding="utf-8"))["notified"] is True


def test_failed_update_check_is_cached(tmp_path: Path) -> None:
    now = datetime(2026, 7, 2, 12, tzinfo=UTC)
    cache_path = tmp_path / "update-check.json"

    def timeout(url: str, *, timeout: float) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    status = refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=cache_path,
        now=now,
        http_get=timeout,
        notice=None,
    )

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["checked_at"] == now.isoformat()
    assert "TimeoutException" in cache["error"]
    assert "TimeoutException" in str(status.error)


def test_update_check_ignores_unwritable_cache_path(tmp_path: Path) -> None:
    status = refresh_update_status(
        "0.3.0",
        PersonalConfig(),
        cache_path=tmp_path,
        http_get=lambda url, *, timeout: FakePyPIResponse("0.3.0"),
        notice=None,
    )

    assert status.latest == "0.3.0"
