from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol

import httpx

from switchboard.app.models.personal import PersonalConfig


@dataclass(frozen=True)
class RuntimeCommandResult:
    ok: bool
    message: str
    model_id: str | None = None


class LocalRuntimeService(Protocol):
    def list_loaded_models(self) -> set[str]:
        ...

    def list_installed_models(self) -> set[str]:
        ...

    def is_model_loaded(self, model_id: str) -> bool:
        ...

    def warm_model(self, model_id: str) -> RuntimeCommandResult:
        ...

    def unload_model(self, model_id: str) -> RuntimeCommandResult:
        ...


class OllamaRuntimeService:
    def __init__(self, config: PersonalConfig) -> None:
        self.config = config
        provider = config.providers.get("ollama")
        self.enabled = bool(provider and provider.enabled)
        self.base_url = (
            provider.base_url if provider and provider.base_url else "http://localhost:11434"
        )
        self.base_url = self.base_url.rstrip("/")

    def list_loaded_models(self) -> set[str]:
        if not self.enabled:
            return set()
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return self._loaded_models_from_api()
        if result.returncode != 0:
            return self._loaded_models_from_api()
        return {_to_model_id(name) for name in _parse_ollama_names(result.stdout)}

    def list_installed_models(self) -> set[str]:
        if not self.enabled:
            return set()
        try:
            result = subprocess.run(
                ["ollama", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return self._installed_models_from_api()
        if result.returncode != 0:
            return self._installed_models_from_api()
        return {_to_model_id(name) for name in _parse_ollama_names(result.stdout)}

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self.list_loaded_models()

    def warm_model(self, model_id: str) -> RuntimeCommandResult:
        if not self.enabled:
            return RuntimeCommandResult(False, "Ollama provider is disabled.", model_id)
        model_name = _from_model_id(model_id)
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": self.config.local_runtime.keep_alive,
                },
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return RuntimeCommandResult(False, f"Ollama unavailable: {exc}", model_id)
        return RuntimeCommandResult(True, f"Warmed {model_id}.", model_id)

    def unload_model(self, model_id: str) -> RuntimeCommandResult:
        if not self.enabled:
            return RuntimeCommandResult(False, "Ollama provider is disabled.", model_id)
        model_name = _from_model_id(model_id)
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": "0",
                },
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return RuntimeCommandResult(False, f"Ollama unavailable: {exc}", model_id)
        return RuntimeCommandResult(True, f"Unloaded {model_id}.", model_id)

    def _loaded_models_from_api(self) -> set[str]:
        try:
            response = httpx.get(f"{self.base_url}/api/ps", timeout=2)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return set()
        return {
            _to_model_id(model.get("name", ""))
            for model in payload.get("models", [])
            if model.get("name")
        }

    def _installed_models_from_api(self) -> set[str]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return set()
        return {
            _to_model_id(model.get("name", ""))
            for model in payload.get("models", [])
            if model.get("name")
        }


def _parse_ollama_names(output: str) -> set[str]:
    names: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name"):
            continue
        names.add(stripped.split()[0])
    return names


def _to_model_id(name: str) -> str:
    return name if name.startswith("ollama/") else f"ollama/{name}"


def _from_model_id(model_id: str) -> str:
    return model_id.split("/", 1)[-1]
