from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class TelemetryRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    request_id: str = SQLField(index=True, unique=True)
    tenant_id: str = SQLField(index=True)
    application_id: str
    workflow_id: str = SQLField(index=True)
    routing_mode: str
    task_type: str
    complexity: str
    sensitivity: str
    classifier_confidence: float
    requested_model: str
    selected_model: str | None = SQLField(default=None, index=True)
    shadow_recommended_model: str | None = None
    policy_version: str
    reason_codes_json: str
    estimated_cost_usd: float = 0.0
    estimated_baseline_cost_usd: float = 0.0
    estimated_latency_ms: int | None = None
    actual_latency_ms: int | None = None
    provider: str | None = None
    fallback_used: bool = False
    status: str = SQLField(index=True)
    error_code: str | None = None
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class TelemetryRead(BaseModel):
    request_id: str
    tenant_id: str
    application_id: str
    workflow_id: str
    routing_mode: str
    task_type: str
    complexity: str
    sensitivity: str
    classifier_confidence: float
    requested_model: str
    selected_model: str | None = None
    shadow_recommended_model: str | None = None
    policy_version: str
    reason_codes: list[str] = Field(default_factory=list)
    estimated_cost_usd: float
    estimated_baseline_cost_usd: float
    estimated_latency_ms: int | None = None
    actual_latency_ms: int | None = None
    provider: str | None = None
    fallback_used: bool
    status: str
    error_code: str | None = None
    created_at: datetime


class PersonalTelemetryRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    request_id: str = SQLField(index=True, unique=True)
    user_id: str = SQLField(index=True)
    project: str = SQLField(index=True)
    mode: str
    task_type: str
    complexity: str
    sensitivity: str
    selected_model: str | None = SQLField(default=None, index=True)
    selected_provider: str | None = None
    route_kind: str
    scarce_model: bool = False
    required_confirmation: bool = False
    called_model: bool = False
    recommended_only: bool = True
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    estimated_premium_units: float = 0.0
    estimated_premium_units_saved: float = 0.0
    router_selected_model: str | None = None
    user_forced_model: str | None = None
    final_selected_model: str | None = None
    override_used: bool = False
    override_reason: str | None = None
    override_safety_blocked: bool = False
    escalation_used: bool = False
    original_request_id: str | None = SQLField(default=None, index=True)
    original_model: str | None = None
    escalated_to_model: str | None = None
    escalation_reason: str | None = None
    manual_recommendation: bool = False
    premium_unit_spent: float = 0.0
    premium_unit_saved: float = 0.0
    estimated_api_cost_saved: float = 0.0
    baseline_model: str | None = None
    baseline_route_kind: str | None = None
    baseline_source: str = "config_default"
    feedback_rating: str | None = None
    selected_model_loaded: bool | None = None
    model_switch_avoided: bool = False
    cold_start_expected: bool = False
    performance_mode: str | None = None
    loaded_local_models_json: str = "[]"
    reason_codes_json: str
    status: str = SQLField(index=True)
    cache_hit: bool = False
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class PersonalTelemetryRead(BaseModel):
    request_id: str
    user_id: str
    project: str
    mode: str
    task_type: str
    complexity: str
    sensitivity: str
    selected_model: str | None = None
    selected_provider: str | None = None
    route_kind: str
    scarce_model: bool
    required_confirmation: bool
    called_model: bool
    recommended_only: bool
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    estimated_premium_units: float
    estimated_premium_units_saved: float
    router_selected_model: str | None = None
    user_forced_model: str | None = None
    final_selected_model: str | None = None
    override_used: bool = False
    override_reason: str | None = None
    override_safety_blocked: bool = False
    escalation_used: bool = False
    original_request_id: str | None = None
    original_model: str | None = None
    escalated_to_model: str | None = None
    escalation_reason: str | None = None
    manual_recommendation: bool = False
    premium_unit_spent: float = 0.0
    premium_unit_saved: float = 0.0
    estimated_api_cost_saved: float = 0.0
    baseline_model: str | None = None
    baseline_route_kind: str | None = None
    baseline_source: str = "config_default"
    feedback_rating: str | None = None
    selected_model_loaded: bool | None = None
    model_switch_avoided: bool = False
    cold_start_expected: bool = False
    performance_mode: str | None = None
    loaded_local_models: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    status: str
    cache_hit: bool = False
    created_at: datetime


class BackendMetricRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    request_id: str = SQLField(index=True, unique=True)
    backend: str = SQLField(index=True)
    selected_model: str | None = SQLField(default=None, index=True)
    project: str = SQLField(index=True)
    prompt_char_count: int
    latency_ms: int
    success: bool = SQLField(index=True)
    error_message: str | None = None
    exit_code: int | None = None
    routing_reason: str | None = None
    cost_type: str
    estimated_cost_usd: float | None = None
    private_mode: bool = False
    metadata_json: str = "{}"
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class BackendMetricRead(BaseModel):
    request_id: str
    backend: str
    selected_model: str | None = None
    project: str
    prompt_char_count: int
    latency_ms: int
    success: bool
    error_message: str | None = None
    exit_code: int | None = None
    routing_reason: str | None = None
    cost_type: str
    estimated_cost_usd: float | None = None
    private_mode: bool
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class MemoryItem(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    project: str = SQLField(index=True)
    title: str = SQLField(index=True)
    content: str
    tags_json: str = "[]"
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class FeedbackExampleRecord(SQLModel, table=True):
    """Full (context, response, verdict) snapshot for a thumbs-down, plus the
    corrected route when the user said "wrong model". Stored only when
    store_feedback_examples is enabled; local-only; purgeable."""

    id: int | None = SQLField(default=None, primary_key=True)
    request_id: str = SQLField(index=True)
    rating: str = SQLField(index=True)  # good | too-weak | wrong-route | bad
    detail: str | None = None  # bad_answer | wrong_model
    corrected_backend: str | None = None  # ollama | codex | claude-code
    prompt: str = ""
    context_text: str = ""
    response_text: str = ""
    route_type: str | None = None
    backend: str | None = None
    confidence: float | None = None
    processed: bool = SQLField(default=False, index=True)
    gate_failed: bool = False
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class RecentContextRecord(SQLModel, table=True):
    """Short-lived map request_id -> assembled context, kept only so a
    thumbs-down can snapshot what the model actually saw. Capped."""

    id: int | None = SQLField(default=None, primary_key=True)
    request_id: str = SQLField(index=True, unique=True)
    context_text: str = ""
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class MemoryEmbeddingRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    memory_id: int = SQLField(index=True, unique=True)
    project: str = SQLField(index=True)
    embedding_model: str
    vector_json: str
    created_at: datetime = SQLField(default_factory=utc_now, index=True)


class RoutingCacheRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    cache_key: str = SQLField(index=True, unique=True)
    project: str = SQLField(index=True)
    mode: str
    route_json: str
    hit_count: int = 0
    created_at: datetime = SQLField(default_factory=utc_now, index=True)
    updated_at: datetime = SQLField(default_factory=utc_now, index=True)


class FeedbackRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    request_id: str = SQLField(index=True)
    rating: str = SQLField(index=True)
    note: str | None = None
    preferred_model: str | None = SQLField(default=None, index=True)
    created_at: datetime = SQLField(default_factory=utc_now, index=True)
