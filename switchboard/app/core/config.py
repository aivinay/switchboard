from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_CONFIG_FILES = (
    "models.yaml",
    "policies.yaml",
    "personal.yaml",
    "personal.example.yaml",
    "router_weights.json",
    "tool_dispatcher_weights.json",
    "sensitivity_weights.json",
)


def user_config_dir() -> Path:
    """Return the per-user config directory used by `switchboard init`."""
    config_home = os.getenv("SWITCHBOARD_CONFIG_HOME") or os.getenv("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home).expanduser() / "switchboard"
    return Path.home() / ".config" / "switchboard"


def packaged_config_path(name: str) -> Path:
    return PACKAGE_CONFIG_DIR / name


def resolve_default_config_path(name: str) -> str:
    """Resolve a default config file for source, user, and wheel installs."""
    for candidate in (Path("config") / name, user_config_dir() / name, packaged_config_path(name)):
        if candidate.exists():
            return str(candidate)
    return str(packaged_config_path(name))


def resolve_config_file(path: str, *, base_dir: Path | None = None) -> str:
    """Resolve a configured file path with package defaults as a final fallback."""
    configured = Path(path).expanduser()
    if configured.is_absolute():
        return str(configured)

    candidates = [configured]
    if base_dir is not None:
        candidates.append(base_dir.expanduser() / configured)
        candidates.append(base_dir.expanduser() / configured.name)
    candidates.extend((user_config_dir() / configured.name, packaged_config_path(configured.name)))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(configured)


class Settings(BaseSettings):
    # `ICP_` is the historical settings prefix, kept for backward compatibility.
    model_config = SettingsConfigDict(env_prefix="ICP_", env_file=".env", extra="ignore")

    environment: str = "local"
    database_url: str = "sqlite:///./switchboard.db"
    models_config_path: str = Field(
        default_factory=lambda: resolve_default_config_path("models.yaml")
    )
    policies_config_path: str = Field(
        default_factory=lambda: resolve_default_config_path("policies.yaml")
    )
    personal_config_path: str = Field(
        default_factory=lambda: resolve_default_config_path("personal.yaml")
    )
    log_prompts: bool = False
    log_responses: bool = False
    request_id_prefix: str = "req"


@lru_cache
def get_settings() -> Settings:
    return Settings()
