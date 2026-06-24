from __future__ import annotations

import os
from pathlib import Path

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.cli_agents import ClaudeCodeCliAdapter, CodexCliAdapter
from switchboard.app.backends.ollama_backend import OllamaAdapter
from switchboard.app.models.backends import BackendInfo
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.local_runtime import OllamaRuntimeService


class BackendRegistry:
    def __init__(self, adapters: dict[str, AgentAdapter]) -> None:
        self.adapters = adapters

    @classmethod
    def default(
        cls,
        container: ServiceContainer,
        *,
        cwd: Path | None = None,
    ) -> BackendRegistry:
        ollama_base_url = (
            container.personal_config.provider_base_url("ollama") or "http://localhost:11434"
        )
        return cls(
            {
                "ollama": OllamaAdapter(
                    catalogue=container.catalogue,
                    provider=OllamaProviderAdapter(
                        ollama_base_url,
                        container.cost_estimator,
                    ),
                    runtime=OllamaRuntimeService(container.personal_config),
                    cost_estimator=container.cost_estimator,
                ),
                "codex": CodexCliAdapter(
                    executable=os.getenv("SWITCHBOARD_CODEX_EXECUTABLE", "codex"),
                    cwd=cwd,
                ),
                "claude-code": ClaudeCodeCliAdapter(
                    executable=os.getenv("SWITCHBOARD_CLAUDE_CODE_EXECUTABLE", "claude"),
                    cwd=cwd,
                    allow_web_search=(
                        container.personal_config.preferences.claude_code_web_search
                    ),
                ),
            }
        )

    def get(self, name: str) -> AgentAdapter | None:
        return self.adapters.get(name)

    def list_backends(self) -> list[BackendInfo]:
        return [adapter.availability() for adapter in self.adapters.values()]

    def available_names(self) -> list[str]:
        return [name for name, adapter in self.adapters.items() if adapter.is_available()]
