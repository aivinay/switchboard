from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class QualityTier(StrEnum):
    EMBEDDING = "embedding"
    SMALL = "small"
    MEDIUM = "medium"
    FRONTIER = "frontier"


class ModelKind(StrEnum):
    LOCAL = "local"
    LOCAL_EMBEDDING = "local_embedding"
    CLOUD_API = "cloud_api"
    MANUAL_SUBSCRIPTION = "manual_subscription"
    MOCK = "mock"
    OPENAI_COMPATIBLE_LOCAL = "openai_compatible_local"


class ModelProfile(BaseModel):
    model_id: str
    provider: str
    provider_model_name: str | None = None
    display_name: str
    kind: ModelKind = ModelKind.MOCK
    context_window: int = 8192
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0
    cost_per_million_input_tokens: float | None = None
    cost_per_million_output_tokens: float | None = None
    average_latency_ms: int = 250
    supports_tools: bool = False
    supports_json_schema: bool = True
    supports_vision: bool = False
    allowed_regions: list[str] = Field(default_factory=list)
    data_policy: str = "local"
    scarce: bool = False
    privacy: str = "local"
    good_for: list[str] = Field(default_factory=list)
    notes: str | None = None
    quality_tier: QualityTier
    enabled: bool = True
    must_never_be_selected_for_chat: bool = False

    @model_validator(mode="after")
    def sync_cost_fields(self) -> ModelProfile:
        if self.provider_model_name is None:
            self.provider_model_name = self.model_id.split("/", 1)[-1]

        if self.cost_per_million_input_tokens is None:
            self.cost_per_million_input_tokens = self.input_cost_per_million_tokens
        else:
            self.input_cost_per_million_tokens = self.cost_per_million_input_tokens

        if self.cost_per_million_output_tokens is None:
            self.cost_per_million_output_tokens = self.output_cost_per_million_tokens
        else:
            self.output_cost_per_million_tokens = self.cost_per_million_output_tokens
        return self

    @property
    def is_private(self) -> bool:
        return self.data_policy in {"private", "mock-private"} or self.privacy == "private"

    @property
    def is_chat_selectable(self) -> bool:
        return (
            self.kind != ModelKind.LOCAL_EMBEDDING
            and not self.must_never_be_selected_for_chat
            and "embeddings" not in self.good_for
        )


class ModelCatalogue(BaseModel):
    models: list[ModelProfile]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelCatalogue:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(models=[ModelProfile(**item) for item in payload.get("models", [])])

    def enabled_models(self) -> list[ModelProfile]:
        return [model for model in self.models if model.enabled]

    def get(self, model_id: str) -> ModelProfile | None:
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def frontier_baseline(self) -> ModelProfile:
        frontiers = [
            model for model in self.enabled_models() if model.quality_tier == QualityTier.FRONTIER
        ]
        if frontiers:
            return max(frontiers, key=lambda model: model.input_cost_per_million_tokens)
        enabled = self.enabled_models()
        if not enabled:
            raise ValueError("model catalogue has no enabled models")
        return max(enabled, key=lambda model: model.input_cost_per_million_tokens)
