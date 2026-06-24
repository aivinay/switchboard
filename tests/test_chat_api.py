from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from switchboard.app.core.config import Settings
from switchboard.app.main import create_app

ROOT = Path(__file__).resolve().parents[1]


def chat_payload(
    content: str,
    model: str = "mock/frontier",
    routing_mode: str = "active",
    tenant_id: str = "demo",
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 120,
        "metadata": {
            "tenant_id": tenant_id,
            "application_id": "tests",
            "workflow_id": "default",
            "environment": "test",
            "routing_mode": routing_mode,
        },
    }


def test_root_points_to_demo_endpoints(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["product"] == "Switchboard"
    assert body["links"]["docs"] == "/docs"
    assert body["links"]["personal_health"] == "/personal/health"


def test_chat_completion_happy_path(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload("Summarise this short ticket in 3 bullets."),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock/small"
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_streaming_returns_clear_error(client: TestClient) -> None:
    payload = chat_payload("Hello")
    payload["stream"] = True

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "STREAMING_NOT_IMPLEMENTED"


def test_observe_mode_logs_shadow_recommendation(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "Summarise this short ticket in 3 bullets.",
            model="mock/frontier",
            routing_mode="observe",
        ),
    )

    assert response.status_code == 200
    assert response.json()["model"] == "mock/frontier"
    requests = client.get("/admin/requests").json()
    record = requests[0]
    assert record["selected_model"] == "mock/frontier"
    assert record["shadow_recommended_model"] == "mock/small"
    assert "OBSERVE_REQUESTED_MODEL_USED" in record["reason_codes"]
    assert "CFO_SHADOW_RECOMMENDATION_LOGGED_FOR_SAVINGS_ANALYSIS" in record["reason_codes"]


def test_active_mode_selects_cheaper_model_for_low_risk_request(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "Summarise this public changelog in 3 bullets.",
            model="mock/frontier",
            routing_mode="active",
        ),
    )

    assert response.status_code == 200
    assert response.json()["model"] == "mock/small"
    record = client.get("/admin/requests").json()[0]
    assert "CFO_SMALL_MODEL_RIGHTSIZED_FOR_SIMPLE_TASK" in record["reason_codes"]
    assert "CFO_LOWEST_COST_POLICY_APPROVED_MODEL_SELECTED" in record["reason_codes"]


def test_coding_request_routes_to_medium_with_explainable_reason_codes(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "Debug this Python code:\n```python\nprint(customer_id)\n```",
            model="mock/frontier",
            routing_mode="active",
        ),
    )

    assert response.status_code == 200
    assert response.json()["model"] == "mock/medium"
    record = client.get("/admin/requests").json()[0]
    assert "CTO_CODING_TASK_DETECTED" in record["reason_codes"]
    assert "CTO_CODING_WORKLOAD_NEEDS_CAPABLE_MODEL" in record["reason_codes"]


def test_demo_policy_denies_regulated_request(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(
            "Prepare a regulated medical risk memo for a patient treatment decision.",
            tenant_id="restricted",
        ),
    )

    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["code"] == "POLICY_DENIED"
    assert "SECURITY_SENSITIVITY_LEVEL_BLOCKED_BY_POLICY" in body["detail"]["reason_codes"]


def test_policy_denial_when_no_model_allowed(tmp_path: Path) -> None:
    policies_path = tmp_path / "policies.yaml"
    policies_path.write_text(
        """
policies:
  - policy_id: deny-all
    tenant_id: default
    workflow_id: default
    version: "1"
    allowed_providers: ["missing"]
    blocked_providers: []
    allowed_models: []
    blocked_models: []
    max_cost_per_request_usd: 0.05
    max_latency_ms: 1500
    allowed_sensitivity_levels: ["public", "internal", "confidential", "regulated", "unknown"]
    require_private_model_for_regulated_data: false
    allow_prompt_logging: false
    allow_response_logging: false
    fallback_model: null
    default_routing_mode: active
""",
        encoding="utf-8",
    )
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path / 'deny.db'}",
        models_config_path=str(ROOT / "config" / "models.yaml"),
        policies_config_path=str(policies_path),
    )
    deny_client = TestClient(create_app(settings))

    response = deny_client.post(
        "/v1/chat/completions",
        json=chat_payload("Summarise this public changelog.", tenant_id="default"),
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "POLICY_DENIED"
    records = deny_client.get("/admin/requests").json()
    assert records[0]["status"] == "denied"
    assert records[0]["error_code"] == "POLICY_DENIED"


def test_telemetry_does_not_store_prompt_or_response_body_by_default(
    client: TestClient,
) -> None:
    sensitive_text = "customer secret ABC-123 should never appear in telemetry"
    response = client.post(
        "/v1/chat/completions",
        json=chat_payload(sensitive_text),
    )

    assert response.status_code == 200
    records = client.get("/admin/requests").json()
    serialized_records = str(records)
    assert sensitive_text not in serialized_records
    assert "messages" not in records[0]
    assert "prompt" not in records[0]
    assert "response" not in records[0]


def test_metrics_endpoints(client: TestClient) -> None:
    client.post(
        "/v1/chat/completions",
        json=chat_payload("Summarise this public changelog in 3 bullets."),
    )

    summary = client.get("/admin/metrics/summary")
    savings = client.get("/admin/metrics/savings")

    assert summary.status_code == 200
    assert savings.status_code == 200
    assert summary.json()["total_requests"] == 1
    assert savings.json()["baseline"] == "everything_goes_to_frontier_model"
