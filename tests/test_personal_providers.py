from __future__ import annotations

import asyncio

import httpx

from switchboard.app.models.api import ChatMessage
from switchboard.app.models.catalogue import ModelKind, ModelProfile, QualityTier
from switchboard.app.models.internal import NormalizedRequest, RoutingMode
from switchboard.app.providers.manual import ManualSubscriptionProviderAdapter
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.utils.time import utc_now

ORIGINAL_OLLAMA_COMPLETE_CHAT = OllamaProviderAdapter.complete_chat


def test_manual_subscription_provider_never_calls_external_service() -> None:
    model = ModelProfile(
        model_id="manual/claude-web",
        provider="claude_web",
        display_name="Claude Web",
        kind=ModelKind.MANUAL_SUBSCRIPTION,
        quality_tier=QualityTier.FRONTIER,
        scarce=True,
        privacy="manual",
    )
    request = NormalizedRequest(
        request_id="req_test",
        tenant_id="local-user",
        application_id="tests",
        workflow_id="personal",
        environment="test",
        messages=[ChatMessage(role="user", content="Plan this architecture.")],
        input_token_estimate=10,
        requested_model="personal/auto",
        metadata={},
        routing_mode=RoutingMode.ACTIVE,
        created_at=utc_now(),
    )

    response = asyncio.run(ManualSubscriptionProviderAdapter().complete_chat(request, model))

    assert response.finish_reason == "recommendation_only"
    assert "does not automate subscription web UIs" in response.content


def test_ollama_provider_uses_provider_model_name(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"content": "local answer"}}

    class AsyncClient:
        def __init__(self, timeout: int) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> AsyncClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> Response:
            captured["url"] = url
            captured["json"] = json
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    monkeypatch.setattr(OllamaProviderAdapter, "complete_chat", ORIGINAL_OLLAMA_COMPLETE_CHAT)
    model = ModelProfile(
        model_id="ollama/gemma4:12b",
        provider="ollama",
        provider_model_name="gemma4:12b",
        display_name="Gemma",
        kind=ModelKind.LOCAL,
        quality_tier=QualityTier.MEDIUM,
    )
    request = NormalizedRequest(
        request_id="req_test",
        tenant_id="local-user",
        application_id="tests",
        workflow_id="personal",
        environment="test",
        messages=[ChatMessage(role="user", content="Summarise this.")],
        input_token_estimate=10,
        requested_model="personal/auto",
        metadata={},
        routing_mode=RoutingMode.ACTIVE,
        created_at=utc_now(),
    )

    response = asyncio.run(OllamaProviderAdapter().complete_chat(request, model))

    assert response.content == "local answer"
    assert captured["json"]["model"] == "gemma4:12b"  # type: ignore[index]
