from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from switchboard.app.core.config import (
    DEFAULT_CONFIG_FILES,
    Settings,
    get_settings,
    packaged_config_path,
)
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.personal import PersonalConfig
from switchboard.app.models.policy import PolicySet
from switchboard.app.services.container import build_container
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.cli import init_command


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_use_packaged_defaults_outside_source_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWITCHBOARD_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in (
        "ICP_MODELS_CONFIG_PATH",
        "ICP_POLICIES_CONFIG_PATH",
        "ICP_PERSONAL_CONFIG_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'switchboard.db'}")

    assert Path(settings.models_config_path).exists()
    assert Path(settings.policies_config_path).exists()
    assert Path(settings.personal_config_path).exists()
    assert "switchboard/config" in settings.models_config_path
    ModelCatalogue.from_yaml(settings.models_config_path)
    PolicySet.from_yaml(settings.policies_config_path)
    PersonalConfig.from_yaml(settings.personal_config_path)


def test_build_container_resolves_bundled_weight_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWITCHBOARD_CONFIG_HOME", str(tmp_path / "xdg"))

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'switchboard.db'}")
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    container = build_container(settings, engine)

    preferences = container.personal_config.preferences
    assert Path(preferences.router_weights_path).exists()
    assert Path(preferences.tool_dispatcher_weights_path).exists()
    assert Path(preferences.sensitivity_weights_path).exists()


def test_init_command_copies_editable_defaults(tmp_path, monkeypatch, capsys) -> None:
    target = tmp_path / "switchboard" / "personal.yaml"
    monkeypatch.setenv("ICP_PERSONAL_CONFIG_PATH", str(target))

    init_command(argparse.Namespace(force=False))

    output = capsys.readouterr().out
    assert "Wrote config:" in output
    assert target.exists()
    for name in DEFAULT_CONFIG_FILES:
        expected = target if name == "personal.yaml" else target.parent / name
        assert expected.exists()


def test_init_command_writes_user_config_when_defaults_are_packaged(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWITCHBOARD_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in (
        "ICP_MODELS_CONFIG_PATH",
        "ICP_POLICIES_CONFIG_PATH",
        "ICP_PERSONAL_CONFIG_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    init_command(argparse.Namespace(force=False))

    config_dir = tmp_path / "xdg" / "switchboard"
    assert (config_dir / "personal.yaml").exists()
    assert (config_dir / "models.yaml").exists()
    assert packaged_config_path("personal.yaml").read_text(encoding="utf-8") == (
        config_dir / "personal.yaml"
    ).read_text(encoding="utf-8")


def test_root_config_tree_matches_packaged_config() -> None:
    root = Path(__file__).resolve().parents[1]
    root_config = root / "config"
    packaged_config = root / "switchboard" / "config"

    config_suffixes = {".json", ".yaml"}
    root_files = sorted(
        path.name
        for path in root_config.iterdir()
        if path.is_file() and path.suffix in config_suffixes
    )
    packaged_files = sorted(
        path.name
        for path in packaged_config.iterdir()
        if path.is_file() and path.suffix in config_suffixes
    )

    assert root_files == packaged_files
    for name in packaged_files:
        assert (root_config / name).read_bytes() == (packaged_config / name).read_bytes()


def test_shipped_personal_configs_contain_safe_defaults() -> None:
    root = Path(__file__).resolve().parents[1]

    for path in (
        root / "config" / "personal.yaml",
        root / "switchboard" / "config" / "personal.yaml",
    ):
        config = PersonalConfig.from_yaml(path)
        text = path.read_text(encoding="utf-8").lower()

        assert config.profile.user_id == "local-user"
        assert "/users/" not in text
        assert "@" not in text
        assert "vinay" not in text
