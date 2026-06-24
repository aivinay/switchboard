from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from switchboard.app.models.api import ChatMessage


class RoutingMode(StrEnum):
    OBSERVE = "observe"
    GUARDED = "guarded"
    ACTIVE = "active"


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARISATION = "summarisation"
    FACTUAL_QA = "factual_qa"
    CODING = "coding"
    DEBUGGING = "debugging"
    ARCHITECTURE_DESIGN = "architecture_design"
    REASONING = "reasoning"
    PLANNING = "planning"
    CREATIVE = "creative"
    REWRITE = "rewrite"
    PRIVATE_SENSITIVE = "private_sensitive"
    AGENTIC_TOOL_USE = "agentic_tool_use"
    UNKNOWN = "unknown"


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    REGULATED = "regulated"
    PRIVATE_PERSONAL = "private_personal"
    UNKNOWN = "unknown"


class LatencyClass(StrEnum):
    INTERACTIVE = "interactive"
    BATCH = "batch"
    UNKNOWN = "unknown"


class NormalizedRequest(BaseModel):
    request_id: str
    tenant_id: str
    application_id: str
    workflow_id: str
    environment: str
    messages: list[ChatMessage]
    input_token_estimate: int
    requested_model: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    routing_mode: RoutingMode
    max_tokens: int | None = None
    temperature: float | None = None
    created_at: datetime


class ClassificationResult(BaseModel):
    task_type: TaskType
    complexity: Complexity
    sensitivity: Sensitivity
    latency_class: LatencyClass
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)


class CostEstimate(BaseModel):
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class RouteDecision(BaseModel):
    selected_model: str
    provider: str
    shadow_recommended_model: str | None = None
    estimated_cost: CostEstimate
    estimated_baseline_cost: CostEstimate
    estimated_latency_ms: int
    reason_codes: list[str] = Field(default_factory=list)
    fallback_used: bool = False
