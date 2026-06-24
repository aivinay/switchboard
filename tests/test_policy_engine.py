from __future__ import annotations

from pathlib import Path

from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.internal import (
    ClassificationResult,
    Complexity,
    LatencyClass,
    Sensitivity,
    TaskType,
)
from switchboard.app.models.policy import TenantPolicy
from switchboard.app.services.cost import CostEstimator
from switchboard.app.services.policy_engine import PolicyEngine

ROOT = Path(__file__).resolve().parents[1]


def classification(sensitivity: Sensitivity = Sensitivity.PUBLIC) -> ClassificationResult:
    return ClassificationResult(
        task_type=TaskType.SUMMARISATION,
        complexity=Complexity.LOW,
        sensitivity=sensitivity,
        latency_class=LatencyClass.INTERACTIVE,
        confidence=0.8,
    )


def test_policy_filters_to_private_model_for_regulated_data() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    policy = TenantPolicy(
        policy_id="test",
        tenant_id="demo",
        version="1",
        allowed_providers=["mock"],
        require_private_model_for_regulated_data=True,
        allowed_sensitivity_levels=[
            Sensitivity.PUBLIC,
            Sensitivity.INTERNAL,
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
        ],
    )

    decision = PolicyEngine(CostEstimator()).evaluate(
        policy,
        classification(Sensitivity.REGULATED),
        catalogue.enabled_models(),
        input_tokens=100,
        output_tokens=100,
    )

    assert decision.allowed
    assert [model.model_id for model in decision.candidate_models] == ["mock/frontier"]
    assert "POLICY_PRIVATE_MODEL_REQUIRED" in decision.reason_codes
    assert "SECURITY_PRIVATE_MODEL_REQUIRED_FOR_REGULATED_DATA" in decision.reason_codes


def test_policy_denies_when_no_candidate_models_remain() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    policy = TenantPolicy(
        policy_id="test",
        tenant_id="demo",
        version="1",
        allowed_providers=["missing"],
        allowed_sensitivity_levels=[Sensitivity.PUBLIC],
    )

    decision = PolicyEngine(CostEstimator()).evaluate(
        policy,
        classification(Sensitivity.PUBLIC),
        catalogue.enabled_models(),
        input_tokens=100,
        output_tokens=100,
    )

    assert not decision.allowed
    assert "NO_POLICY_ALLOWED_MODELS" in decision.reason_codes
    assert "SECURITY_NO_POLICY_APPROVED_MODEL_AVAILABLE" in decision.reason_codes
