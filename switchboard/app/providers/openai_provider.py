from __future__ import annotations

import os

import httpx

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.providers.base import ProviderAdapter, ProviderResponse
from switchboard.app.services.cost import CostEstimator


class OpenAIProviderAdapter(ProviderAdapter):
    provider_name = "openai"

    def __init__(self, cost_estimator: CostEstimator | None = None) -> None:
        self.cost_estimator = cost_estimator or CostEstimator()

    async def complete_chat(
        self,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        payload = {
            "model": model_profile.model_id.split("/", 1)[-1],
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={key: value for key, value in payload.items() if value is not None},
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return ProviderResponse(
            content=content,
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=usage.get("prompt_tokens", request.input_token_estimate),
            completion_tokens=usage.get(
                "completion_tokens", self.cost_estimator.estimate_text_tokens(content)
            ),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )
