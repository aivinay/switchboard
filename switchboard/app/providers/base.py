from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import NormalizedRequest


class ProviderResponse(BaseModel):
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class ProviderAdapter(ABC):
    provider_name: str

    @abstractmethod
    async def complete_chat(
        self,
        request: NormalizedRequest,
        model_profile: ModelProfile,
    ) -> ProviderResponse:
        raise NotImplementedError
