from __future__ import annotations

from time import perf_counter

from fastapi import status

from switchboard.app.core.errors import http_error
from switchboard.app.models.api import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from switchboard.app.models.internal import NormalizedRequest, RoutingMode
from switchboard.app.services.container import ServiceContainer
from switchboard.app.utils.ids import new_request_id
from switchboard.app.utils.time import unix_timestamp, utc_now


class ChatCompletionService:
    def __init__(self, container: ServiceContainer) -> None:
        self.container = container

    async def complete(self, payload: ChatCompletionRequest) -> ChatCompletionResponse:
        normalized = self._normalize(payload)
        policy = self.container.policies.match(
            tenant_id=normalized.tenant_id,
            workflow_id=normalized.workflow_id,
        )
        classification = self.container.classifier.classify(normalized)
        output_tokens = self.container.cost_estimator.expected_output_tokens(normalized)
        policy_decision = self.container.policy_engine.evaluate(
            policy=policy,
            classification=classification,
            candidate_models=self.container.catalogue.enabled_models(),
            input_tokens=normalized.input_token_estimate,
            output_tokens=output_tokens,
        )

        if not policy_decision.allowed:
            self.container.telemetry.record_denied(
                normalized,
                classification,
                policy_decision,
                error_code="POLICY_DENIED",
            )
            raise http_error(
                status.HTTP_403_FORBIDDEN,
                "POLICY_DENIED",
                "No allowed model route is available for this request.",
                [*classification.reason_codes, *policy_decision.reason_codes],
            )

        route = self.container.router.route(normalized, classification, policy, policy_decision)
        model_profile = self.container.catalogue.get(route.selected_model)
        if model_profile is None:
            raise http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "SELECTED_MODEL_NOT_FOUND",
                f"Selected model is missing from catalogue: {route.selected_model}",
                route.reason_codes,
            )

        adapter = self.container.providers.get(route.provider)
        if adapter is None:
            raise http_error(
                status.HTTP_502_BAD_GATEWAY,
                "PROVIDER_ADAPTER_NOT_CONFIGURED",
                f"No provider adapter is configured for provider: {route.provider}",
                route.reason_codes,
            )

        started = perf_counter()
        provider_response = await adapter.complete_chat(normalized, model_profile)
        actual_latency_ms = int((perf_counter() - started) * 1000)
        self.container.telemetry.record_success(
            normalized,
            classification,
            policy_decision,
            route,
            actual_latency_ms=actual_latency_ms,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl_{normalized.request_id}",
            created=unix_timestamp(),
            model=route.selected_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=provider_response.content),
                    finish_reason=provider_response.finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=provider_response.prompt_tokens,
                completion_tokens=provider_response.completion_tokens,
                total_tokens=provider_response.prompt_tokens + provider_response.completion_tokens,
            ),
            system_fingerprint="switchboard-local",
        )

    def _normalize(self, payload: ChatCompletionRequest) -> NormalizedRequest:
        metadata = payload.metadata or {}
        tenant_id = str(metadata.get("tenant_id") or "default")
        application_id = str(metadata.get("application_id") or "default")
        workflow_id = str(metadata.get("workflow_id") or "default")
        environment = str(metadata.get("environment") or self.container.settings.environment)
        policy = self.container.policies.match(tenant_id=tenant_id, workflow_id=workflow_id)

        routing_mode_value = str(
            metadata.get("routing_mode") or policy.default_routing_mode
        ).lower()
        if routing_mode_value not in {mode.value for mode in RoutingMode}:
            raise http_error(
                status.HTTP_400_BAD_REQUEST,
                "INVALID_ROUTING_MODE",
                f"Unsupported routing mode: {routing_mode_value}",
                ["INVALID_ROUTING_MODE"],
            )

        input_tokens = self.container.cost_estimator.estimate_text_tokens(
            "\n".join(message.content for message in payload.messages)
        )
        return NormalizedRequest(
            request_id=new_request_id(self.container.settings.request_id_prefix),
            tenant_id=tenant_id,
            application_id=application_id,
            workflow_id=workflow_id,
            environment=environment,
            messages=payload.messages,
            input_token_estimate=input_tokens,
            requested_model=payload.model,
            metadata=metadata,
            routing_mode=RoutingMode(routing_mode_value),
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            created_at=utc_now(),
        )
