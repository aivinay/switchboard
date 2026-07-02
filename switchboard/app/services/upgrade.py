from __future__ import annotations

import importlib.metadata
import json
import shlex
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from switchboard.app.services.update_check import PACKAGE_NAME

_DIRECT_URL_UNSET = object()


@dataclass(frozen=True)
class UpgradePlan:
    install_method: str
    command: tuple[str, ...]
    can_execute: bool
    reason: str

    @property
    def command_text(self) -> str:
        return shlex.join(self.command)


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part.lower() for part in path.expanduser().parts)


def _is_pipx_prefix(prefix: Path) -> bool:
    parts = _path_parts(prefix)
    return "pipx" in parts and "venvs" in parts


def _is_uv_tool_prefix(prefix: Path) -> bool:
    parts = _path_parts(prefix)
    return "uv" in parts and "tools" in parts


def _read_direct_url_text() -> str | None:
    try:
        return importlib.metadata.distribution(PACKAGE_NAME).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_editable_install(direct_url_text: str | None) -> bool:
    if not direct_url_text:
        return False
    try:
        payload = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    dir_info = payload.get("dir_info")
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


def _git_checkout_root(package_file: Path) -> Path | None:
    repo_root = package_file.resolve().parent.parent
    return repo_root if (repo_root / ".git").exists() else None


def _externally_managed() -> bool:
    stdlib = sysconfig.get_path("stdlib")
    return bool(stdlib and (Path(stdlib) / "EXTERNALLY-MANAGED").exists())


def _manual_plan(install_method: str, command: tuple[str, ...], reason: str) -> UpgradePlan:
    return UpgradePlan(
        install_method=install_method,
        command=command,
        can_execute=False,
        reason=reason,
    )


def detect_upgrade_plan(
    *,
    prefix: Path | None = None,
    base_prefix: Path | None = None,
    executable: str | None = None,
    package_file: Path | None = None,
    direct_url_text: str | None | object = _DIRECT_URL_UNSET,
    externally_managed: bool | None = None,
) -> UpgradePlan:
    resolved_prefix = (prefix or Path(sys.prefix)).resolve()
    resolved_base = (base_prefix or Path(sys.base_prefix)).resolve()
    resolved_executable = executable or sys.executable
    direct_url = (
        _read_direct_url_text()
        if direct_url_text is _DIRECT_URL_UNSET
        else direct_url_text
    )
    direct_url_value = direct_url if isinstance(direct_url, str) else None
    package_path = package_file or Path(__file__).resolve()
    repo_root = _git_checkout_root(package_path)

    if _is_editable_install(direct_url_value):
        root = repo_root or package_path.resolve().parent.parent
        return _manual_plan(
            "editable",
            (
                "sh",
                "-c",
                f"cd {shlex.quote(str(root))} && "
                f"{shlex.quote(resolved_executable)} -m pip install -e .",
            ),
            "Editable installs should be upgraded from the checkout.",
        )
    if repo_root is not None:
        return _manual_plan(
            "git-checkout",
            (
                "sh",
                "-c",
                f"cd {shlex.quote(str(repo_root))} && git pull && "
                f"{shlex.quote(resolved_executable)} -m pip install -e .",
            ),
            "Source checkouts should be updated with git, then reinstalled.",
        )
    is_externally_managed = (
        externally_managed if externally_managed is not None else _externally_managed()
    )
    if is_externally_managed:
        return _manual_plan(
            "externally-managed",
            ("pipx", "install", "--force", PACKAGE_NAME),
            "This Python is externally managed; use an app-managed environment.",
        )
    if _is_pipx_prefix(resolved_prefix):
        return UpgradePlan(
            install_method="pipx",
            command=("pipx", "upgrade", PACKAGE_NAME),
            can_execute=True,
            reason="Detected a pipx-managed application environment.",
        )
    if _is_uv_tool_prefix(resolved_prefix):
        return UpgradePlan(
            install_method="uv-tool",
            command=("uv", "tool", "upgrade", PACKAGE_NAME),
            can_execute=True,
            reason="Detected a uv tool environment.",
        )
    if resolved_prefix != resolved_base:
        return UpgradePlan(
            install_method="venv-pip",
            command=(resolved_executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME),
            can_execute=True,
            reason="Detected a virtual environment or plain pip install.",
        )
    return UpgradePlan(
        install_method="pip",
        command=(resolved_executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME),
        can_execute=True,
        reason="Detected a plain pip install.",
    )
