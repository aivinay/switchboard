from __future__ import annotations

import asyncio
from pathlib import Path

from switchboard.app.models.api import ChatMessage
from switchboard.app.models.catalogue import ModelCatalogue
from switchboard.app.models.internal import NormalizedRequest, RoutingMode
from switchboard.app.providers.mock import MockProviderAdapter
from switchboard.app.utils.time import utc_now

ROOT = Path(__file__).resolve().parents[1]


def test_mock_provider_returns_deterministic_response() -> None:
    catalogue = ModelCatalogue.from_yaml(ROOT / "config" / "models.yaml")
    model = catalogue.get("mock/small")
    assert model is not None
    request = NormalizedRequest(
        request_id="req_test",
        tenant_id="demo",
        application_id="tests",
        workflow_id="default",
        environment="test",
        messages=[ChatMessage(role="user", content="Hello")],
        input_token_estimate=2,
        requested_model="mock/small",
        metadata={},
        routing_mode=RoutingMode.ACTIVE,
        created_at=utc_now(),
    )

    response = asyncio.run(MockProviderAdapter().complete_chat(request, model))

    assert response.model == "mock/small"
    assert response.provider == "mock"
    assert response.content.startswith("Demo mock response only from mock/small")
    assert "Enable Ollama or LM Studio for real local answers" in response.content
    assert "workflow default" in response.content
