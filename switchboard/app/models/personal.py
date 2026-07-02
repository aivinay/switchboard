from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from switchboard.app.models.catalogue import ModelKind, ModelProfile


class PersonalProfile(BaseModel):
    user_id: str = "local-user"
    default_project: str = "personal"


class PersonalPreferences(BaseModel):
    default_mode: str = "auto"
    local_first: bool = True
    prefer_free_models: bool = True
    allow_cloud: bool = False
    require_confirmation_for_scarce_models: bool = True
    private_mode: bool = True
    avoid_scarce_for_simple_tasks: bool = True
    use_coding_model_for_coding: bool = True
    use_frontier_or_manual_recommendation_for_complex_reasoning: bool = True
    compress_long_context_before_premium_recommendation: bool = True
    cache_routing: bool = True
    cache_answers: bool = False
    # Switchboard Core router mode: "rules" (Phase A deterministic), "llm"
    # (local LLM router with rules fallback), "hybrid" (rules first, LLM only
    # for ambiguous prompts), or "learned" (trained embedding classifier with
    # rules fallback).
    router_mode: str = "rules"
    llm_router_model: str = "llama3.2:3b"
    # Trained-router weights file (relative to repo root) and confidence floor.
    router_weights_path: str = "config/router_weights.json"
    learned_router_min_confidence: float = 0.55
    # Learned tool dispatcher: second-chance tool detection when the regex
    # detector finds nothing. Predictions only count after the actual tool
    # verifies them. Active whenever enabled AND trained weights exist
    # (train with `switchboard train-dispatcher`).
    tool_dispatcher_enabled: bool = True
    tool_dispatcher_weights_path: str = "config/tool_dispatcher_weights.json"
    # 0.8 keeps measured false positives near the regex detector's level
    # while still recovering most missed recall (CLINC150 held-out sweep).
    tool_dispatcher_min_confidence: float = 0.8
    # Learned sensitivity escalator: catches private phrasings the keyword
    # hints miss. Can only ADD protection (keyword positives are final);
    # active whenever enabled AND trained weights exist
    # (train with `switchboard train-sensitivity`).
    sensitivity_escalator_enabled: bool = True
    sensitivity_weights_path: str = "config/sensitivity_weights.json"
    sensitivity_escalator_min_confidence: float = 0.7
    # Headroom-style heuristic prompt compression before routing.
    compression_enabled: bool = False
    compression_threshold_tokens: int = 1000
    # Embedding-based long-term semantic memory (local embeddings only).
    semantic_memory_enabled: bool = False
    semantic_memory_top_k: int = 3
    embedding_model: str = "nomic-embed-text"
    # Optional local model role mappings. Values are catalogue model IDs and
    # are only honored when the model is enabled, local, and chat-selectable.
    local_model_roles: dict[str, str] = Field(default_factory=dict)
    # Allow the Claude Code adapter to use its WebSearch tool (pre-approved via
    # --allowedTools). Off by default; uses your Claude subscription's search.
    claude_code_web_search: bool = False
    # Live-data providers, enabled by default with keyless services so every
    # install grounds stock/news answers out of the box. Empty values fall back
    # to env-based provider defaults; use "none" to disable a provider.
    # Time, date, calculator, and unit conversion are always on (no provider
    # needed); web search stays off until a Brave API key is configured.
    finance_provider: str = "yahoo"
    news_provider: str = "google_news_rss"
    # Closed feedback loop: store full (context, response) snapshots for
    # thumbs-downs and auto-retrain the learned router once enough corrected
    # "wrong model" examples accumulate. Local-only; purge with
    # `switchboard feedback-examples --purge`.
    store_feedback_examples: bool = False
    feedback_retrain_threshold: int = 5
    project_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PersonalBudgets(BaseModel):
    monthly_api_budget_usd: float = 10.0
    daily_premium_units: int = 20


class PersonalSavingsConfig(BaseModel):
    default_baseline_model: str = "manual/claude-web"
    premium_unit_value_usd: float | None = None
    assume_premium_for_unknown: bool = False


class LocalRuntimeConfig(BaseModel):
    performance_mode: str = "balanced"
    max_loaded_models: int = 2
    keep_alive: str = "10m"
    reuse_hot_model_if_good_enough: bool = True
    model_switch_penalty_ms: int = 3000
    prefer_hot_model_for_simple_tasks: bool = True
    unload_after_benchmark: bool = True


class PersonalProviderConfig(BaseModel):
    type: ModelKind | str
    enabled: bool = False
    base_url: str | None = None
    env_api_key: str | None = None
    scarce: bool = False
    notes: str | None = None


class PersonalConfig(BaseModel):
    profile: PersonalProfile = Field(default_factory=PersonalProfile)
    preferences: PersonalPreferences = Field(default_factory=PersonalPreferences)
    budgets: PersonalBudgets = Field(default_factory=PersonalBudgets)
    savings: PersonalSavingsConfig = Field(default_factory=PersonalSavingsConfig)
    local_runtime: LocalRuntimeConfig = Field(default_factory=LocalRuntimeConfig)
    providers: dict[str, PersonalProviderConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PersonalConfig:
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(**payload)

    def provider_enabled(self, provider: str) -> bool:
        provider_config = self.providers.get(provider)
        return bool(provider_config and provider_config.enabled)

    def provider_type(self, provider: str) -> str | None:
        provider_config = self.providers.get(provider)
        return str(provider_config.type) if provider_config else None

    def provider_base_url(self, provider: str) -> str | None:
        provider_config = self.providers.get(provider)
        return provider_config.base_url if provider_config else None


class PersonalPromptRequest(BaseModel):
    prompt: str
    project: str | None = None
    mode: str | None = None
    use_cache: bool = True
    strict: bool = False
    force_model: str | None = None
    allow_cloud_once: bool = False
    override_reason: str | None = None
    baseline_model: str | None = None
    original_request_id: str | None = None
    escalation_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressionResult(BaseModel):
    original_estimated_tokens: int
    compressed_estimated_tokens: int
    compression_used: bool
    estimated_tokens_saved: int
    compression_ratio: float = 1.0
    compressed_prompt: str | None = None
    warning: str | None = None
    # Which part of the input was compressible: "history_only" for assembled
    # session contexts (fact blocks preserved verbatim), "whole_text" for raw
    # prompts, "none" below threshold, None for the legacy compress() path.
    scope: str | None = None


class PremiumPrompt(BaseModel):
    title: str
    recommended_tool: str
    ready_to_paste_prompt: str
    why_this_tool: str
    what_to_try_locally_first: str
    estimated_tokens_saved: int


class PersonalRouteResponse(BaseModel):
    request_id: str
    user_id: str
    project: str
    mode: str
    task_type: str
    complexity: str
    sensitivity: str
    selected_model: str
    selected_provider: str
    recommended_model: str
    recommended_provider: str
    route_kind: str
    scarce_model: bool
    requires_confirmation: bool
    called_model: bool = False
    recommended_only: bool = True
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    estimated_premium_units: float
    estimated_premium_units_saved: float
    compression: CompressionResult
    confidence: float
    uncertainty_reasons: list[str] = Field(default_factory=list)
    privacy_note: str
    next_best_alternative: str | None = None
    premium_prompt: PremiumPrompt | None = None
    cache_hit: bool = False
    explanation: str
    reason_codes: list[str] = Field(default_factory=list)
    router_selected_model: str | None = None
    user_forced_model: str | None = None
    final_selected_model: str | None = None
    override_used: bool = False
    override_reason: str | None = None
    override_safety_blocked: bool = False
    escalation_used: bool = False
    original_request_id: str | None = None
    baseline_model: str | None = None
    baseline_route_kind: str | None = None
    baseline_source: str = "config_default"
    premium_unit_spent: float = 0.0
    premium_unit_saved: float = 0.0
    estimated_api_cost_saved: float = 0.0
    performance_mode: str | None = None
    selected_model_loaded: bool | None = None
    model_switch_avoided: bool = False
    cold_start_expected: bool = False
    loaded_local_models: list[str] = Field(default_factory=list)


class PersonalAskResponse(BaseModel):
    request_id: str
    answer: str | None = None
    recommendation: PersonalRouteResponse
    suggested_compressed_prompt: str | None = None
    quality_warning: bool = False
    quality_notes: list[str] = Field(default_factory=list)
    suggested_next_step: str | None = None
    status: str


class PersonalModelRead(BaseModel):
    model_id: str
    provider_model_name: str | None = None
    provider: str
    display_name: str
    kind: str
    quality_tier: str
    scarce: bool
    privacy: str
    enabled: bool
    provider_enabled: bool
    good_for: list[str] = Field(default_factory=list)
    notes: str | None = None
    must_never_be_selected_for_chat: bool = False

    @classmethod
    def from_profile(cls, model: ModelProfile, provider_enabled: bool) -> PersonalModelRead:
        return cls(
            model_id=model.model_id,
            provider_model_name=model.provider_model_name,
            provider=model.provider,
            display_name=model.display_name,
            kind=model.kind.value,
            quality_tier=model.quality_tier.value,
            scarce=model.scarce,
            privacy=model.privacy,
            enabled=model.enabled,
            provider_enabled=provider_enabled,
            good_for=model.good_for,
            notes=model.notes,
            must_never_be_selected_for_chat=model.must_never_be_selected_for_chat,
        )


class PersonalMemoryCreate(BaseModel):
    title: str
    content: str
    project: str | None = None
    tags: list[str] = Field(default_factory=list)


class PersonalMemoryRead(BaseModel):
    id: int
    project: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: str


class FeedbackCreate(BaseModel):
    request_id: str
    rating: str
    note: str | None = None
    preferred_model: str | None = None


class FeedbackRead(BaseModel):
    request_id: str
    rating: str
    note: str | None = None
    preferred_model: str | None = None
    created_at: str
