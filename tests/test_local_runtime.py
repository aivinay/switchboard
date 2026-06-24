from __future__ import annotations

import subprocess
from typing import Any

import httpx

from switchboard.app.models.personal import PersonalConfig, PersonalProviderConfig
from switchboard.app.services.local_runtime import OllamaRuntimeService


def ollama_config() -> PersonalConfig:
    return PersonalConfig(
        providers={
            "ollama": PersonalProviderConfig(
                type="local",
                enabled=True,
                base_url="http://localhost:11434",
            )
        }
    )


def test_list_loaded_models_parses_ollama_ps(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = (
            "NAME            ID              SIZE      PROCESSOR    UNTIL\n"
            "qwen3:8b        abc123          5.2 GB    100% GPU     9 minutes\n"
            "llama3.2:3b     def456          2.0 GB    100% GPU     9 minutes\n"
        )

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Result())

    loaded = OllamaRuntimeService(ollama_config()).list_loaded_models()

    assert loaded == {"ollama/qwen3:8b", "ollama/llama3.2:3b"}


def test_list_loaded_models_handles_unavailable_ollama(monkeypatch) -> None:
    def missing_ollama(*args, **kwargs):  # noqa: ANN002, ANN003
        raise FileNotFoundError("ollama")

    monkeypatch.setattr(subprocess, "run", missing_ollama)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )

    loaded = OllamaRuntimeService(ollama_config()).list_loaded_models()

    assert loaded == set()


def test_warm_and_unload_model_use_ollama_generate_endpoint(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url: str, json: dict[str, object], timeout: int) -> Response:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(httpx, "post", fake_post)

    service = OllamaRuntimeService(ollama_config())
    warm = service.warm_model("ollama/qwen3:8b")
    unload = service.unload_model("ollama/qwen3:8b")

    assert warm.ok
    assert unload.ok
    assert calls[0]["url"] == "http://localhost:11434/api/generate"
    assert calls[0]["json"]["model"] == "qwen3:8b"
    assert calls[0]["json"]["keep_alive"] == "10m"
    assert calls[1]["json"]["keep_alive"] == "0"
