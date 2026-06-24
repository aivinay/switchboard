from __future__ import annotations

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.internal import ClassificationResult, Sensitivity
from switchboard.app.models.policy import PolicyDecision, TenantPolicy
from switchboard.app.services.cost import CostEstimator


class PolicyEngine:
    def __init__(self, cost_estimator: CostEstimator) -> None:
        self.cost_estimator = cost_estimator

    def evaluate(
        self,
        policy: TenantPolicy,
        classification: ClassificationResult,
        candidate_models: list[ModelProfile],
        input_tokens: int,
        output_tokens: int,
    ) -> PolicyDecision:
        reason_codes: list[str] = []

        if (
            policy.allowed_sensitivity_levels
            and classification.sensitivity not in policy.allowed_sensitivity_levels
        ):
            return PolicyDecision(
                allowed=False,
                reason_codes=[
                    "SENSITIVITY_LEVEL_BLOCKED",
                    "SECURITY_SENSITIVITY_LEVEL_BLOCKED_BY_POLICY",
                ],
                candidate_models=[],
                policy_version=policy.version,
                policy_id=policy.policy_id,
            )

        filtered: list[ModelProfile] = []
        for model in candidate_models:
            if policy.allowed_providers and model.provider not in policy.allowed_providers:
                continue
            if model.provider in policy.blocked_providers:
                continue
            if policy.allowed_models and model.model_id not in policy.allowed_models:
                continue
            if model.model_id in policy.blocked_models:
                continue
            if (
                policy.max_latency_ms is not None
                and model.average_latency_ms > policy.max_latency_ms
            ):
                continue
            if (
                policy.require_private_model_for_regulated_data
                and classification.sensitivity == Sensitivity.REGULATED
                and not model.is_private
            ):
                continue
            if policy.max_cost_per_request_usd is not None:
                estimate = self.cost_estimator.estimate(model, input_tokens, output_tokens)
                if estimate.total_cost_usd > policy.max_cost_per_request_usd:
                    continue
            filtered.append(model)

        if policy.allowed_providers:
            reason_codes.extend(
                ["POLICY_ALLOWED_PROVIDERS_APPLIED", "SECURITY_PROVIDER_ALLOWLIST_ENFORCED"]
            )
        if policy.allowed_models:
            reason_codes.extend(
                ["POLICY_ALLOWED_MODELS_APPLIED", "SECURITY_MODEL_ALLOWLIST_ENFORCED"]
            )
        if policy.blocked_providers or policy.blocked_models:
            reason_codes.extend(["POLICY_BLOCKLIST_APPLIED", "SECURITY_BLOCKLIST_ENFORCED"])
        if policy.max_cost_per_request_usd is not None:
            reason_codes.extend(["POLICY_MAX_COST_APPLIED", "CFO_REQUEST_COST_CAP_ENFORCED"])
        if policy.max_latency_ms is not None:
            reason_codes.extend(["POLICY_MAX_LATENCY_APPLIED", "CTO_LATENCY_SLO_ENFORCED"])
        if (
            policy.require_private_model_for_regulated_data
            and classification.sensitivity == Sensitivity.REGULATED
        ):
            reason_codes.extend(
                [
                    "POLICY_PRIVATE_MODEL_REQUIRED",
                    "SECURITY_PRIVATE_MODEL_REQUIRED_FOR_REGULATED_DATA",
                ]
            )

        if not filtered:
            reason_codes.extend(
                ["NO_POLICY_ALLOWED_MODELS", "SECURITY_NO_POLICY_APPROVED_MODEL_AVAILABLE"]
            )
            return PolicyDecision(
                allowed=False,
                reason_codes=reason_codes,
                candidate_models=[],
                policy_version=policy.version,
                policy_id=policy.policy_id,
            )

        reason_codes.extend(["POLICY_ALLOWED_CANDIDATES", "GOVERNANCE_POLICY_APPROVED_CANDIDATES"])
        return PolicyDecision(
            allowed=True,
            reason_codes=reason_codes,
            candidate_models=filtered,
            policy_version=policy.version,
            policy_id=policy.policy_id,
        )
