from __future__ import annotations

from switchboard.app.models.catalogue import ModelKind, ModelProfile
from switchboard.app.models.internal import NormalizedRequest
from switchboard.app.providers.base import ProviderAdapter, ProviderResponse
from switchboard.app.services.cost import CostEstimator


class MockProviderAdapter(ProviderAdapter):
    provider_name = "mock"

    def __init__(self, cost_estimator: CostEstimator | None = None) -> None:
        self.cost_estimator = cost_estimator or CostEstimator()

    async def complete_chat(
        self,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        if model_profile.kind == ModelKind.MOCK:
            content = (
                f"Demo mock response only from {model_profile.model_id}. "
                "Enable Ollama or LM Studio for real local answers. "
                f"Received {len(request.messages)} message(s) for workflow {request.workflow_id}."
            )
        else:
            content = (
                f"Mock response from {model_profile.model_id}: "
                f"received {len(request.messages)} message(s) for workflow {request.workflow_id}."
            )
        return ProviderResponse(
            content=content,
            model=model_profile.model_id,
            provider=model_profile.provider,
            prompt_tokens=request.input_token_estimate,
            completion_tokens=self.cost_estimator.estimate_text_tokens(content),
        )
