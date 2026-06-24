from __future__ import annotations

from switchboard.app.providers.anthropic_provider import AnthropicProviderAdapter
from switchboard.app.providers.base import ProviderAdapter
from switchboard.app.providers.lmstudio import LMStudioProviderAdapter
from switchboard.app.providers.manual import ManualSubscriptionProviderAdapter
from switchboard.app.providers.mock import MockProviderAdapter
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.providers.openai_provider import OpenAIProviderAdapter
from switchboard.app.services.cost import CostEstimator


class ProviderRegistry:
    def __init__(self, adapters: dict[str, ProviderAdapter]) -> None:
        self.adapters = adapters

    @classmethod
    def default(
        cls,
        cost_estimator: CostEstimator,
        ollama_base_url: str = "http://localhost:11434",
        lmstudio_base_url: str = "http://localhost:1234/v1",
    ) -> ProviderRegistry:
        return cls(
            {
                "mock": MockProviderAdapter(cost_estimator),
                "ollama": OllamaProviderAdapter(ollama_base_url, cost_estimator),
                "lmstudio": LMStudioProviderAdapter(lmstudio_base_url, cost_estimator),
                "openai": OpenAIProviderAdapter(cost_estimator),
                "anthropic": AnthropicProviderAdapter(cost_estimator),
                "claude_web": ManualSubscriptionProviderAdapter(),
                "chatgpt_web": ManualSubscriptionProviderAdapter(),
                "codex": ManualSubscriptionProviderAdapter(),
            }
        )

    def get(self, provider: str) -> ProviderAdapter | None:
        return self.adapters.get(provider)
