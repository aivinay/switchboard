from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from switchboard.app.core.config import Settings
from switchboard.app.main import create_app
from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.providers.anthropic_provider import AnthropicProviderAdapter
from switchboard.app.providers.base import ProviderResponse
from switchboard.app.providers.lmstudio import LMStudioProviderAdapter
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.providers.openai_provider import OpenAIProviderAdapter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    personal_config = tmp_path / "personal_test.yaml"
    personal_config.write_text(
        """
profile:
  user_id: "local-user"
  default_project: "personal"
preferences:
  default_mode: "auto"
  local_first: true
  prefer_free_models: true
  allow_cloud: false
  require_confirmation_for_scarce_models: true
  private_mode: true
  cache_routing: true
  cache_answers: false
  # Tests must stay offline and deterministic: live-data providers are
  # enabled by default in production, and learned assists may call the local
  # embedder when trained weights exist, so pin them off here explicitly.
  router_mode: "rules"
  tool_dispatcher_enabled: false
  sensitivity_escalator_enabled: false
  compression_enabled: false
  semantic_memory_enabled: false
  store_feedback_examples: false
  finance_provider: ""
  news_provider: ""
providers:
  mock:
    type: "mock"
    enabled: true
  ollama:
    type: "local"
    base_url: "http://localhost:11434"
    enabled: false
  claude_web:
    type: "manual_subscription"
    enabled: true
    scarce: true
  chatgpt_web:
    type: "manual_subscription"
    enabled: true
    scarce: true
  codex:
    type: "manual_subscription"
    enabled: true
    scarce: true
""",
        encoding="utf-8",
    )
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'switchboard_test.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(ROOT / "config" / "policies.yaml"),
        personal_config_path=str(personal_config),
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    return TestClient(create_app(test_settings))


@pytest.fixture(autouse=True)
def block_real_provider_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_if_called(
        self: object,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        raise AssertionError(f"Real provider adapter was called in tests: {model_profile.provider}")

    monkeypatch.setattr(OpenAIProviderAdapter, "complete_chat", fail_if_called)
    monkeypatch.setattr(AnthropicProviderAdapter, "complete_chat", fail_if_called)
    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", fail_if_called)
    monkeypatch.setattr(LMStudioProviderAdapter, "complete_chat", fail_if_called)


def chat_payload(
    content: str,
    model: str = "mock/frontier",
    routing_mode: str = "active",
    tenant_id: str = "demo",
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 120,
        "metadata": {
            "tenant_id": tenant_id,
            "application_id": "tests",
            "workflow_id": "default",
            "environment": "test",
            "routing_mode": routing_mode,
        },
    }
