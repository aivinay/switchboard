from __future__ import annotations

import httpx

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.providers.base import ProviderAdapter, ProviderResponse
from switchboard.app.services.cost import CostEstimator


class OllamaProviderAdapter(ProviderAdapter):
    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cost_estimator = cost_estimator or CostEstimator()

    async def complete_chat(
        self,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        payload = {
            "model": model_profile.provider_model_name or model_profile.model_id.split("/", 1)[-1],
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.base_url}: {type(exc).__name__}: {exc}"
            ) from exc

        content = data.get("message", {}).get("content", "")
        return ProviderResponse(
            content=content,
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=self.cost_estimator.estimate_text_tokens(content),
        )
