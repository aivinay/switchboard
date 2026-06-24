from __future__ import annotations

import json

from switchboard.app.models.internal import (
    ClassificationResult,
    NormalizedRequest,
    RouteDecision,
)
from switchboard.app.models.policy import PolicyDecision
from switchboard.app.models.telemetry import TelemetryRecord
from switchboard.app.storage.repositories import TelemetryRepository


class TelemetryService:
    def __init__(self, repository: TelemetryRepository) -> None:
        self.repository = repository

    def record_success(
        self,
        request: NormalizedRequest,
        classification: ClassificationResult,
        policy_decision: PolicyDecision,
        route: RouteDecision,
        actual_latency_ms: int,
    ) -> TelemetryRecord:
        return self.repository.add(
            TelemetryRecord(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                application_id=request.application_id,
                workflow_id=request.workflow_id,
                routing_mode=request.routing_mode.value,
                task_type=classification.task_type.value,
                complexity=classification.complexity.value,
                sensitivity=classification.sensitivity.value,
                classifier_confidence=classification.confidence,
                requested_model=request.requested_model,
                selected_model=route.selected_model,
                shadow_recommended_model=route.shadow_recommended_model,
                policy_version=policy_decision.policy_version,
                reason_codes_json=json.dumps(route.reason_codes),
                estimated_cost_usd=route.estimated_cost.total_cost_usd,
                estimated_baseline_cost_usd=route.estimated_baseline_cost.total_cost_usd,
                estimated_latency_ms=route.estimated_latency_ms,
                actual_latency_ms=actual_latency_ms,
                provider=route.provider,
                fallback_used=route.fallback_used,
                status="success",
            )
        )

    def record_denied(
        self,
        request: NormalizedRequest,
        classification: ClassificationResult,
        policy_decision: PolicyDecision,
        error_code: str,
    ) -> TelemetryRecord:
        reason_codes = [*classification.reason_codes, *policy_decision.reason_codes]
        return self.repository.add(
            TelemetryRecord(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                application_id=request.application_id,
                workflow_id=request.workflow_id,
                routing_mode=request.routing_mode.value,
                task_type=classification.task_type.value,
                complexity=classification.complexity.value,
                sensitivity=classification.sensitivity.value,
                classifier_confidence=classification.confidence,
                requested_model=request.requested_model,
                selected_model=None,
                shadow_recommended_model=None,
                policy_version=policy_decision.policy_version,
                reason_codes_json=json.dumps(reason_codes),
                status="denied",
                error_code=error_code,
            )
        )
