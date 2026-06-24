from __future__ import annotations

from pathlib import Path

from switchboard.app.models.api import ChatMessage
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.internal import (
    ClassificationResult,
    Complexity,
    LatencyClass,
    NormalizedRequest,
    RoutingMode,
    Sensitivity,
    TaskType,
)
from switchboard.app.models.policy import PolicyDecision, TenantPolicy
from switchboard.app.services.cost import CostEstimator
from switchboard.app.services.router import RoutingEngine
from switchboard.app.utils.time import utc_now

ROOT = Path(__file__).resolve().parents[1]


def make_request(mode: RoutingMode = RoutingMode.ACTIVE) -> NormalizedRequest:
    return NormalizedRequest(
        request_id="req_test",
        tenant_id="demo",
        application_id="tests",
        workflow_id="default",
        environment="test",
        messages=[ChatMessage(role="user", content="Summarise this short ticket.")],
        input_token_estimate=20,
        requested_model="mock/frontier",
        metadata={},
        routing_mode=mode,
        max_tokens=120,
        created_at=utc_now(),
    )


def make_policy() -> TenantPolicy:
    return TenantPolicy(
        policy_id="test",
        tenant_id="demo",
        version="1",
        allowed_providers=["mock"],
        fallback_model="mock/frontier",
        allowed_sensitivity_levels=[
            Sensitivity.PUBLIC,
            Sensitivity.INTERNAL,
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
        ],
    )


def make_decision(catalogue: ModelCatalogue) -> PolicyDecision:
    return PolicyDecision(
        allowed=True,
        reason_codes=["POLICY_ALLOWED_CANDIDATES"],
        candidate_models=catalogue.enabled_models(),
        policy_version="1",
        policy_id="test",
    )


def test_active_low_complexity_summary_selects_small_model() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    route = RoutingEngine(catalogue, CostEstimator()).route(
        make_request(),
        ClassificationResult(
            task_type=TaskType.SUMMARISATION,
            complexity=Complexity.LOW,
            sensitivity=Sensitivity.PUBLIC,
            latency_class=LatencyClass.INTERACTIVE,
            confidence=0.8,
        ),
        make_policy(),
        make_decision(catalogue),
    )

    assert route.selected_model == "mock/small"
    assert "LOW_COMPLEXITY_SUMMARY" in route.reason_codes
    assert "CFO_SMALL_MODEL_RIGHTSIZED_FOR_SIMPLE_TASK" in route.reason_codes
    assert "ACTIVE_ROUTER_USED" in route.reason_codes


def test_observe_mode_uses_requested_model_and_logs_shadow_route() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    route = RoutingEngine(catalogue, CostEstimator()).route(
        make_request(RoutingMode.OBSERVE),
        ClassificationResult(
            task_type=TaskType.SUMMARISATION,
            complexity=Complexity.LOW,
            sensitivity=Sensitivity.PUBLIC,
            latency_class=LatencyClass.INTERACTIVE,
            confidence=0.8,
        ),
        make_policy(),
        make_decision(catalogue),
    )

    assert route.selected_model == "mock/frontier"
    assert route.shadow_recommended_model == "mock/small"
    assert "OBSERVE_REQUESTED_MODEL_USED" in route.reason_codes
    assert "CFO_SHADOW_RECOMMENDATION_LOGGED_FOR_SAVINGS_ANALYSIS" in route.reason_codes


def test_regulated_request_selects_frontier_private_model() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    route = RoutingEngine(catalogue, CostEstimator()).route(
        make_request(),
        ClassificationResult(
            task_type=TaskType.REASONING,
            complexity=Complexity.HIGH,
            sensitivity=Sensitivity.REGULATED,
            latency_class=LatencyClass.INTERACTIVE,
            confidence=0.7,
        ),
        make_policy(),
        PolicyDecision(
            allowed=True,
            reason_codes=["POLICY_PRIVATE_MODEL_REQUIRED"],
            candidate_models=[catalogue.get("mock/frontier")],
            policy_version="1",
            policy_id="test",
        ),
    )

    assert route.selected_model == "mock/frontier"
    assert "REGULATED_DOMAIN" in route.reason_codes
    assert "SECURITY_REGULATED_ROUTE_REQUIRES_STRONG_MODEL" in route.reason_codes
