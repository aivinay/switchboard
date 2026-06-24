from __future__ import annotations

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.providers.base import ProviderAdapter, ProviderResponse


class ManualSubscriptionProviderAdapter(ProviderAdapter):
    provider_name = "manual_subscription"

    async def complete_chat(
        self,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        content = (
            f"Recommendation only: use {model_profile.display_name} manually for this task. "
            "Switchboard does not automate subscription web UIs or bypass provider limits."
        )
        return ProviderResponse(
            content=content,
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=0,
            finish_reason="recommendation_only",
        )
