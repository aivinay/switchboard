from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)


class CliAgentAdapter(AgentAdapter):
    def __init__(
        self,
        *,
        name: str,
        executable: str,
        cost_type: BackendCostType = BackendCostType.SUBSCRIPTION,
        cwd: Path | None = None,
    ) -> None:
        self.name = name
        self.executable = executable
        self.cost_type = cost_type
        self.cwd = cwd

    def executable_path(self) -> str | None:
        return shutil.which(self.executable)

    def is_available(self) -> bool:
        return self.executable_path() is not None

    def availability(self) -> BackendInfo:
        path = self.executable_path()
        return BackendInfo(
            name=self.name,
            available=path is not None,
            path=path,
            cost_type=self.cost_type,
            warning=None if path else f"{self.executable} was not found on PATH.",
        )

    def build_command(self, request: SwitchboardRequest) -> list[str]:
        raise NotImplementedError

    def selected_model(self, request: SwitchboardRequest) -> str:
        return request.model or f"{self.name}/default"

    def response_content(self, stdout: str, success: bool) -> str | None:
        return stdout.strip() if success else None

    def response_model(self, request: SwitchboardRequest, stdout: str) -> str:
        return self.selected_model(request)

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        selected_model = self.selected_model(request)
        if not self.is_available():
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                selected_model=selected_model,
                success=False,
                error_message=f"{self.name} CLI is unavailable: {self.executable} not found.",
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )

        command = self.build_command(request)
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                cwd=str(self.cwd) if self.cwd else None,
            )
        except subprocess.TimeoutExpired as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                selected_model=selected_model,
                stdout=stdout,
                stderr=stderr,
                latency_ms=latency_ms,
                success=False,
                error_message=f"{self.name} timed out after {request.timeout_s}s.",
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        success = result.returncode == 0
        selected_model = self.response_model(request, stdout)
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            selected_model=selected_model,
            content=self.response_content(stdout, success),
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
            latency_ms=latency_ms,
            success=success,
            error_message=None
            if success
            else (stderr.strip() or f"{self.name} exited with code {result.returncode}."),
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )


class CodexCliAdapter(CliAgentAdapter):
    def __init__(
        self,
        executable: str = "codex",
        cwd: Path | None = None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__(name="codex", executable=executable, cwd=cwd)
        self.config_path = config_path

    def build_command(self, request: SwitchboardRequest) -> list[str]:
        command = [
            self.executable_path() or self.executable,
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            # Allow running outside a git repository (e.g. zip checkouts,
            # benchmark runs). The read-only sandbox above still applies.
            "--skip-git-repo-check",
        ]
        if self.cwd:
            command.extend(["--cd", str(self.cwd)])
        if request.model:
            command.extend(["--model", request.model])
        command.append(request.prompt)
        return command

    def selected_model(self, request: SwitchboardRequest) -> str:
        if request.model:
            return request.model
        return self.configured_default_model() or "codex/default"

    def configured_default_model(self) -> str | None:
        config_path = self.config_path or Path(
            os.getenv("CODEX_CONFIG", "")
            or Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser() / "config.toml"
        )
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None
        model = payload.get("model")
        return model if isinstance(model, str) and model else None


class ClaudeCodeCliAdapter(CliAgentAdapter):
    def __init__(
        self,
        executable: str = "claude",
        cwd: Path | None = None,
        *,
        allow_web_search: bool = False,
    ) -> None:
        super().__init__(name="claude-code", executable=executable, cwd=cwd)
        self.allow_web_search = allow_web_search

    def build_command(self, request: SwitchboardRequest) -> list[str]:
        command = [
            self.executable_path() or self.executable,
            "--print",
            "--output-format=json",
            "--no-session-persistence",
            "--permission-mode",
            "default",
            "--disallowedTools=Edit,Write,Bash",
        ]
        if self.allow_web_search:
            # Pre-approve WebSearch so non-interactive runs never stall asking
            # for a permission the user cannot grant.
            command.append("--allowedTools=WebSearch")
        if request.model:
            command.extend(["--model", request.model])
        command.append(request.prompt)
        return command

    def response_content(self, stdout: str, success: bool) -> str | None:
        if not success:
            return None
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout.strip()
        result = payload.get("result")
        return result if isinstance(result, str) else stdout.strip()

    def response_model(self, request: SwitchboardRequest, stdout: str) -> str:
        if request.model:
            return request.model
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return self.selected_model(request)
        model_usage = payload.get("modelUsage")
        if not isinstance(model_usage, dict) or not model_usage:
            return self.selected_model(request)
        first_key = next(iter(model_usage))
        if not isinstance(first_key, str) or not first_key:
            return self.selected_model(request)
        return first_key.split("[", 1)[0]
