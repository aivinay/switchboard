from __future__ import annotations

from switchboard.app.models.catalogue import ModelCatalogue, ModelProfile, QualityTier
from switchboard.app.models.internal import (
    ClassificationResult,
    Complexity,
    NormalizedRequest,
    RouteDecision,
    RoutingMode,
    Sensitivity,
    TaskType,
)
from switchboard.app.models.policy import PolicyDecision, TenantPolicy
from switchboard.app.services.cost import CostEstimator

TIER_RANK = {
    QualityTier.EMBEDDING: -1,
    QualityTier.SMALL: 0,
    QualityTier.MEDIUM: 1,
    QualityTier.FRONTIER: 2,
}


class RoutingEngine:
    def __init__(self, catalogue: ModelCatalogue, cost_estimator: CostEstimator) -> None:
        self.catalogue = catalogue
        self.cost_estimator = cost_estimator

    def route(
        self,
        request: NormalizedRequest,
        classification: ClassificationResult,
        policy: TenantPolicy,
        policy_decision: PolicyDecision,
    ) -> RouteDecision:
        candidates = policy_decision.candidate_models
        input_tokens = request.input_token_estimate
        output_tokens = self.cost_estimator.expected_output_tokens(request)

        recommended, route_reasons = self._recommend_model(classification, candidates, policy)
        selected = recommended
        shadow_recommended_model: str | None = recommended.model_id
        fallback_used = False
        mode_reason: list[str] = []

        if request.routing_mode == RoutingMode.OBSERVE:
            requested = self._candidate_by_id(candidates, request.requested_model)
            if requested is not None:
                selected = requested
                mode_reason.extend(
                    [
                        "OBSERVE_REQUESTED_MODEL_USED",
                        "CFO_SHADOW_RECOMMENDATION_LOGGED_FOR_SAVINGS_ANALYSIS",
                    ]
                )
            else:
                selected = self._safest_model(candidates, policy) or recommended
                fallback_used = selected.model_id != recommended.model_id
                mode_reason.extend(
                    [
                        "OBSERVE_REQUESTED_MODEL_BLOCKED",
                        "FALLBACK_MODEL_USED",
                        "SECURITY_REQUESTED_MODEL_NOT_POLICY_APPROVED",
                    ]
                )

        elif request.routing_mode == RoutingMode.GUARDED:
            if self._guarded_allows_routing(classification):
                selected = recommended
                mode_reason.extend(["GUARDED_ROUTER_USED", "GOVERNANCE_GUARDED_ROUTING_ALLOWED"])
            else:
                selected = self._safest_model(candidates, policy) or recommended
                fallback_used = selected.model_id != recommended.model_id
                mode_reason.extend(
                    ["GUARDED_FALLBACK_USED", "SECURITY_GUARDED_MODE_ESCALATED_TO_SAFE_MODEL"]
                )

        else:
            mode_reason.extend(["ACTIVE_ROUTER_USED", "GOVERNANCE_ACTIVE_ROUTING_APPLIED"])

        selected_cost = self.cost_estimator.estimate(selected, input_tokens, output_tokens)
        baseline_model = self.catalogue.frontier_baseline()
        baseline_cost = self.cost_estimator.estimate(baseline_model, input_tokens, output_tokens)

        return RouteDecision(
            selected_model=selected.model_id,
            provider=selected.provider,
            shadow_recommended_model=shadow_recommended_model,
            estimated_cost=selected_cost,
            estimated_baseline_cost=baseline_cost,
            estimated_latency_ms=selected.average_latency_ms,
            reason_codes=[
                *classification.reason_codes,
                *policy_decision.reason_codes,
                *route_reasons,
                *mode_reason,
            ],
            fallback_used=fallback_used,
        )

    def _recommend_model(
        self,
        classification: ClassificationResult,
        candidates: list[ModelProfile],
        policy: TenantPolicy,
    ) -> tuple[ModelProfile, list[str]]:
        target_tier, reason_codes = self._target_tier(classification)
        matching = [model for model in candidates if model.quality_tier == target_tier]

        if matching:
            return self._lowest_cost_latency(matching), [
                *reason_codes,
                "SELECTED_CHEAPEST_ALLOWED_TIER",
                "CFO_LOWEST_COST_POLICY_APPROVED_MODEL_SELECTED",
            ]

        fallback = self._candidate_by_id(candidates, policy.fallback_model)
        if fallback is not None:
            return fallback, [
                *reason_codes,
                "POLICY_FALLBACK_MODEL_USED",
                "GOVERNANCE_POLICY_FALLBACK_MODEL_SELECTED",
            ]

        target_rank = TIER_RANK[target_tier]
        ranked = sorted(
            candidates,
            key=lambda model: (
                abs(TIER_RANK[model.quality_tier] - target_rank),
                model.input_cost_per_million_tokens + model.output_cost_per_million_tokens,
                model.average_latency_ms,
            ),
        )
        return ranked[0], [
            *reason_codes,
            "SELECTED_NEAREST_ALLOWED_MODEL",
            "GOVERNANCE_NEAREST_POLICY_APPROVED_MODEL_SELECTED",
        ]

    def _target_tier(self, classification: ClassificationResult) -> tuple[QualityTier, list[str]]:
        reasons: list[str] = []
        if classification.sensitivity == Sensitivity.REGULATED:
            reasons.extend(["REGULATED_DOMAIN", "SECURITY_REGULATED_ROUTE_REQUIRES_STRONG_MODEL"])
            return QualityTier.FRONTIER, reasons

        if classification.confidence < 0.55:
            reasons.extend(["LOW_CLASSIFIER_CONFIDENCE", "SECURITY_LOW_CONFIDENCE_ESCALATION"])
            return QualityTier.FRONTIER, reasons

        if classification.task_type == TaskType.CODING:
            reasons.extend(["CODING_TASK", "CTO_CODING_WORKLOAD_NEEDS_CAPABLE_MODEL"])
            tier = (
                QualityTier.FRONTIER
                if classification.complexity == Complexity.HIGH
                else QualityTier.MEDIUM
            )
            return tier, reasons

        if (
            classification.complexity == Complexity.HIGH
            or classification.task_type == TaskType.REASONING
        ):
            reasons.extend(
                ["HIGH_COMPLEXITY_REASONING", "CTO_COMPLEX_REASONING_NEEDS_FRONTIER_MODEL"]
            )
            return QualityTier.FRONTIER, reasons

        if (
            classification.task_type
            in {
                TaskType.CLASSIFICATION,
                TaskType.EXTRACTION,
                TaskType.SUMMARISATION,
            }
            and classification.complexity == Complexity.LOW
        ):
            reasons.extend(["LOW_COMPLEXITY_SUMMARY", "CFO_SMALL_MODEL_RIGHTSIZED_FOR_SIMPLE_TASK"])
            return QualityTier.SMALL, reasons

        if classification.task_type in {TaskType.FACTUAL_QA, TaskType.UNKNOWN}:
            reasons.extend(["MEDIUM_COMPLEXITY_QA", "CTO_MEDIUM_MODEL_RIGHTSIZED_FOR_QA"])
            return QualityTier.MEDIUM, reasons

        reasons.extend(["DEFAULT_MEDIUM_ROUTE", "GOVERNANCE_DEFAULT_MEDIUM_MODEL_ROUTE"])
        return QualityTier.MEDIUM, reasons

    def _lowest_cost_latency(self, candidates: list[ModelProfile]) -> ModelProfile:
        return min(
            candidates,
            key=lambda model: (
                model.input_cost_per_million_tokens + model.output_cost_per_million_tokens,
                model.average_latency_ms,
            ),
        )

    def _candidate_by_id(
        self, candidates: list[ModelProfile], model_id: str | None
    ) -> ModelProfile | None:
        if model_id is None:
            return None
        for model in candidates:
            if model.model_id == model_id:
                return model
        return None

    def _safest_model(
        self, candidates: list[ModelProfile], policy: TenantPolicy
    ) -> ModelProfile | None:
        fallback = self._candidate_by_id(candidates, policy.fallback_model)
        if fallback is not None:
            return fallback
        frontier = [model for model in candidates if model.quality_tier == QualityTier.FRONTIER]
        if frontier:
            return self._lowest_cost_latency(frontier)
        return (
            max(candidates, key=lambda model: TIER_RANK[model.quality_tier]) if candidates else None
        )

    def _guarded_allows_routing(self, classification: ClassificationResult) -> bool:
        return (
            classification.confidence >= 0.7
            and classification.sensitivity in {Sensitivity.PUBLIC, Sensitivity.INTERNAL}
            and classification.complexity != Complexity.HIGH
        )
