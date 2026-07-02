"""Switchboard package."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path


def _version_from_pyproject(repo_root: Path) -> str | None:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    version = payload.get("project", {}).get("version")
    return version if isinstance(version, str) and version else None


def _resolve_version(
    *,
    distribution_name: str = "switchboard-local",
    repo_root: Path | None = None,
) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        source_root = repo_root or Path(__file__).resolve().parents[1]
        return _version_from_pyproject(source_root) or "unknown (source checkout)"


__version__ = _resolve_version()
