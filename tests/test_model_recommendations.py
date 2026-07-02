from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from switchboard.app.core.config import Settings
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.personal import PersonalConfig
from switchboard.app.services.model_recommendations import (
    GIB,
    apply_local_model_pack,
    parse_linux_meminfo_total_bytes,
    parse_sysctl_memsize_bytes,
    ram_tier,
    recommend_local_model_pack,
)
from switchboard.cli import models_command

ROOT = Path(__file__).resolve().parents[1]


def test_ram_detection_parsers() -> None:
    assert parse_linux_meminfo_total_bytes("MemTotal:       16384000 kB\n") == 16_384_000 * 1024
    assert parse_sysctl_memsize_bytes("34359738368\n") == 34_359_738_368
    assert parse_sysctl_memsize_bytes("not-a-number\n") is None


def test_ram_tier_mapping() -> None:
    assert ram_tier(None) == "16gb"
    assert ram_tier(16 * GIB) == "16gb"
    assert ram_tier(32 * GIB) == "32gb"
    assert ram_tier(64 * GIB) == "48gb"


def test_recommendations_exclude_disabled_and_embedding_models_for_chat_roles() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")

    recommendation = recommend_local_model_pack(catalogue, total_ram_bytes=64 * GIB)

    chat_roles = [role for role in recommendation.roles if role.role != "embeddings"]
    assert {role.role for role in chat_roles} == {"general", "coding", "reasoning"}
    for role in chat_roles:
        model = catalogue.get(role.model_id)
        assert model is not None
        assert model.enabled
        assert model.is_chat_selectable
    assert "ollama/gemma4:31b" not in {role.model_id for role in chat_roles}
    assert "ollama/embeddinggemma" not in {role.model_id for role in chat_roles}


def test_apply_recommendation_rewrites_local_role_mappings(tmp_path: Path) -> None:
    personal_path = tmp_path / "personal.yaml"
    models_path = tmp_path / "models.yaml"
    personal_path.write_text((ROOT / "config" / "personal.yaml").read_text(encoding="utf-8"))
    models_path.write_text((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    catalogue = ModelCatalogue.from_yaml(models_path)
    recommendation = recommend_local_model_pack(catalogue, total_ram_bytes=32 * GIB)

    apply_local_model_pack(
        personal_config_path=personal_path,
        models_config_path=models_path,
        recommendation=recommendation,
    )

    config = PersonalConfig.from_yaml(personal_path)
    assert config.preferences.local_model_roles["general"] == "ollama/gemma4:12b"
    assert config.preferences.local_model_roles["coding"] == "ollama/qwen3.5:9b"
    assert config.preferences.local_model_roles["reasoning"] == "ollama/gpt-oss:20b"
    assert config.preferences.embedding_model == "embeddinggemma"

    models_payload = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    enabled_by_id = {model["model_id"]: model["enabled"] for model in models_payload["models"]}
    assert enabled_by_id["ollama/gemma4:12b"]
    assert enabled_by_id["ollama/qwen3.5:9b"]
    assert enabled_by_id["ollama/gpt-oss:20b"]


def test_models_recommend_cli_prints_pull_commands(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    personal_path = tmp_path / "personal.yaml"
    models_path = tmp_path / "models.yaml"
    personal_path.write_text((ROOT / "config" / "personal.yaml").read_text(encoding="utf-8"))
    models_path.write_text((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'switchboard.db'}",
        personal_config_path=str(personal_path),
        models_config_path=str(models_path),
    )

    monkeypatch.setattr("switchboard.cli.get_settings", lambda: settings)
    monkeypatch.setattr("switchboard.cli.detect_total_ram_bytes", lambda: 32 * GIB)

    models_command(argparse.Namespace(recommend=True, apply=False, yes=False))

    output = capsys.readouterr().out
    assert "Detected RAM: 32.0 GiB (32gb tier)" in output
    assert "general" in output
    assert "ollama pull gemma4:12b" in output
    assert "No models were pulled automatically." in output
