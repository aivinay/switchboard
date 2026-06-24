from __future__ import annotations

import os

import httpx

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.providers.base import ProviderAdapter, ProviderResponse
from switchboard.app.services.cost import CostEstimator


class AnthropicProviderAdapter(ProviderAdapter):
    provider_name = "anthropic"

    def __init__(self, cost_estimator: CostEstimator | None = None) -> None:
        self.cost_estimator = cost_estimator or CostEstimator()

    async def complete_chat(
        self,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")

        system_messages = [
            message.content for message in request.messages if message.role == "system"
        ]
        non_system_messages = [
            message.model_dump(exclude_none=True)
            for message in request.messages
            if message.role != "system"
        ]
        payload = {
            "model": model_profile.model_id.split("/", 1)[-1],
            "max_tokens": request.max_tokens or 256,
            "temperature": request.temperature,
            "system": "\n".join(system_messages) if system_messages else None,
            "messages": non_system_messages,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={key: value for key, value in payload.items() if value is not None},
            )
            response.raise_for_status()
            data = response.json()

        content = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage") or {}
        return ProviderResponse(
            content=content,
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=usage.get("input_tokens", request.input_token_estimate),
            completion_tokens=usage.get(
                "output_tokens", self.cost_estimator.estimate_text_tokens(content)
            ),
            finish_reason=data.get("stop_reason", "stop"),
        )
