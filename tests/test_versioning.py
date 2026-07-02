from __future__ import annotations

import importlib.metadata
from pathlib import Path

from switchboard import _resolve_version


def test_resolve_version_uses_installed_distribution(monkeypatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.8.7")

    assert _resolve_version(repo_root=Path("/does/not/matter")) == "9.8.7"


def test_resolve_version_reads_pyproject_for_source_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    def missing_distribution(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "switchboard-local"
version = "1.2.3"
""",
        encoding="utf-8",
    )

    assert _resolve_version(repo_root=tmp_path) == "1.2.3"


def test_resolve_version_reports_unknown_without_distribution_or_pyproject(
    monkeypatch, tmp_path: Path
) -> None:
    def missing_distribution(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing_distribution)

    assert _resolve_version(repo_root=tmp_path) == "unknown (source checkout)"
