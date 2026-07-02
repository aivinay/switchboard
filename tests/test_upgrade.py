from __future__ import annotations

from pathlib import Path

from switchboard.app.services.upgrade import detect_upgrade_plan


def test_detect_upgrade_plan_for_pipx() -> None:
    plan = detect_upgrade_plan(
        prefix=Path("/Users/test/.local/pipx/venvs/switchboard-local"),
        base_prefix=Path("/usr/local/python"),
        executable="/venv/bin/python",
        package_file=Path("/venv/lib/site-packages/switchboard/__init__.py"),
        direct_url_text=None,
        externally_managed=False,
    )

    assert plan.install_method == "pipx"
    assert plan.command == ("pipx", "upgrade", "switchboard-local")
    assert plan.can_execute is True


def test_detect_upgrade_plan_for_uv_tool() -> None:
    plan = detect_upgrade_plan(
        prefix=Path("/Users/test/.local/share/uv/tools/switchboard-local"),
        base_prefix=Path("/usr/local/python"),
        executable="/venv/bin/python",
        package_file=Path("/venv/lib/site-packages/switchboard/__init__.py"),
        direct_url_text=None,
        externally_managed=False,
    )

    assert plan.install_method == "uv-tool"
    assert plan.command == ("uv", "tool", "upgrade", "switchboard-local")
    assert plan.can_execute is True


def test_detect_upgrade_plan_for_plain_venv() -> None:
    plan = detect_upgrade_plan(
        prefix=Path("/tmp/project/.venv"),
        base_prefix=Path("/usr/local/python"),
        executable="/tmp/project/.venv/bin/python",
        package_file=Path("/tmp/project/.venv/lib/site-packages/switchboard/__init__.py"),
        direct_url_text=None,
        externally_managed=False,
    )

    assert plan.install_method == "venv-pip"
    assert plan.command == (
        "/tmp/project/.venv/bin/python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "switchboard-local",
    )
    assert plan.can_execute is True


def test_detect_upgrade_plan_for_venv_with_externally_managed_base() -> None:
    plan = detect_upgrade_plan(
        prefix=Path("/tmp/project/.venv"),
        base_prefix=Path("/opt/homebrew"),
        executable="/tmp/project/.venv/bin/python",
        package_file=Path("/tmp/project/.venv/lib/site-packages/switchboard/__init__.py"),
        direct_url_text=None,
        externally_managed=True,
    )

    assert plan.install_method == "venv-pip"
    assert plan.can_execute is True


def test_detect_upgrade_plan_for_editable_install(tmp_path: Path) -> None:
    package_file = tmp_path / "src" / "switchboard" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    direct_url = '{"url": "file:///tmp/switchboard", "dir_info": {"editable": true}}'

    plan = detect_upgrade_plan(
        prefix=Path("/tmp/project/.venv"),
        base_prefix=Path("/usr/local/python"),
        executable="/tmp/project/.venv/bin/python",
        package_file=package_file,
        direct_url_text=direct_url,
        externally_managed=False,
    )

    assert plan.install_method == "editable"
    assert plan.can_execute is False
    assert "pip install -e" in plan.command_text


def test_detect_upgrade_plan_for_git_checkout(tmp_path: Path) -> None:
    package_file = tmp_path / "switchboard" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    (tmp_path / ".git").mkdir()

    plan = detect_upgrade_plan(
        prefix=Path("/tmp/project/.venv"),
        base_prefix=Path("/usr/local/python"),
        executable="/tmp/project/.venv/bin/python",
        package_file=package_file,
        direct_url_text=None,
        externally_managed=False,
    )

    assert plan.install_method == "git-checkout"
    assert plan.can_execute is False
    assert "git pull" in plan.command_text
    assert "make install" in plan.command_text
    assert "pip install -e" not in plan.command_text


def test_detect_upgrade_plan_walks_nested_package_path_to_repo_root(tmp_path: Path) -> None:
    package_file = tmp_path / "switchboard" / "app" / "services" / "upgrade.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='switchboard-local'\n",
        encoding="utf-8",
    )
    (tmp_path / ".git").mkdir()

    plan = detect_upgrade_plan(
        prefix=Path("/tmp/project/.venv"),
        base_prefix=Path("/usr/local/python"),
        executable="/tmp/project/.venv/bin/python",
        package_file=package_file,
        direct_url_text=None,
        externally_managed=False,
    )

    assert plan.install_method == "git-checkout"
    assert plan.can_execute is False
    assert plan.command == ("sh", "-c", f"cd {tmp_path} && git pull && make install")


def test_detect_upgrade_plan_for_editable_checkout_uses_project_install(
    tmp_path: Path,
) -> None:
    package_file = tmp_path / "switchboard" / "app" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='switchboard-local'\n",
        encoding="utf-8",
    )
    (tmp_path / ".git").mkdir()
    direct_url = '{"url": "file:///tmp/switchboard", "dir_info": {"editable": true}}'

    plan = detect_upgrade_plan(
        prefix=Path("/tmp/project/.venv"),
        base_prefix=Path("/usr/local/python"),
        executable="/tmp/project/.venv/bin/python",
        package_file=package_file,
        direct_url_text=direct_url,
        externally_managed=False,
    )

    assert plan.install_method == "editable"
    assert plan.can_execute is False
    assert plan.command == ("sh", "-c", f"cd {tmp_path} && git pull && make install")


def test_detect_upgrade_plan_for_externally_managed_python() -> None:
    plan = detect_upgrade_plan(
        prefix=Path("/opt/homebrew"),
        base_prefix=Path("/opt/homebrew"),
        executable="/opt/homebrew/bin/python3.13",
        package_file=Path("/opt/homebrew/lib/site-packages/switchboard/__init__.py"),
        direct_url_text=None,
        externally_managed=True,
    )

    assert plan.install_method == "externally-managed"
    assert plan.can_execute is False
    assert plan.command == ("pipx", "install", "--force", "switchboard-local")
