from __future__ import annotations

import asyncio
import time

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.models.api import ChatMessage
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    SwitchboardRequest,
    SwitchboardResponse,
)
from switchboard.app.models.catalogue import ModelCatalogue, ModelKind, ModelProfile
from switchboard.app.models.internal import NormalizedRequest, RoutingMode
from switchboard.app.providers.ollama import OllamaProviderAdapter
from switchboard.app.services.cost import CostEstimator
from switchboard.app.services.local_runtime import OllamaRuntimeService


class OllamaAdapter(AgentAdapter):
    name = "ollama"
    cost_type = BackendCostType.LOCAL

    def __init__(
        self,
        *,
        catalogue: ModelCatalogue,
        provider: OllamaProviderAdapter,
        runtime: OllamaRuntimeService,
        cost_estimator: CostEstimator,
    ) -> None:
        self.catalogue = catalogue
        self.provider = provider
        self.runtime = runtime
        self.cost_estimator = cost_estimator

    def is_available(self) -> bool:
        return self.runtime.enabled and bool(self.runtime.list_installed_models())

    def availability(self) -> BackendInfo:
        if not self.runtime.enabled:
            return BackendInfo(
                name=self.name,
                available=False,
                cost_type=self.cost_type,
                warning="Ollama provider is disabled.",
            )
        installed = sorted(self.runtime.list_installed_models())
        return BackendInfo(
            name=self.name,
            available=bool(installed),
            cost_type=self.cost_type,
            details=", ".join(installed) if installed else None,
            warning=None if installed else "Ollama is enabled but no installed models were found.",
        )

    def ask(self, request: SwitchboardRequest) -> SwitchboardResponse:
        started = time.perf_counter()
        model = self._select_model(request.model)
        if model is None:
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                latency_ms=int((time.perf_counter() - started) * 1000),
                success=False,
                error_message="No enabled Ollama chat model is available.",
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )
        normalized = NormalizedRequest(
            request_id=request.request_id,
            tenant_id="local-user",
            application_id="switchboard-core",
            workflow_id=request.project,
            environment="local",
            messages=[ChatMessage(role="user", content=request.prompt)],
            input_token_estimate=self.cost_estimator.estimate_text_tokens(request.prompt),
            requested_model=model.model_id,
            metadata=request.metadata,
            routing_mode=RoutingMode.ACTIVE,
            created_at=request.created_at,
        )
        try:
            provider_response = asyncio.run(self.provider.complete_chat(normalized, model))
        except RuntimeError as exc:
            return SwitchboardResponse(
                request_id=request.request_id,
                backend=self.name,
                selected_model=model.model_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                success=False,
                error_message=str(exc),
                cost_type=self.cost_type,
                estimated_cost_usd=0.0,
            )
        content = provider_response.content
        return SwitchboardResponse(
            request_id=request.request_id,
            backend=self.name,
            content=content,
            selected_model=model.model_id,
            stdout=content,
            latency_ms=int((time.perf_counter() - started) * 1000),
            success=True,
            cost_type=self.cost_type,
            estimated_cost_usd=0.0,
        )

    def _select_model(self, requested_model: str | None) -> ModelProfile | None:
        if requested_model:
            model = self.catalogue.get(requested_model)
            if model and model.provider == "ollama" and model.enabled and model.is_chat_selectable:
                return model
            return None
        for model in self.catalogue.models:
            if (
                model.provider == "ollama"
                and model.kind == ModelKind.LOCAL
                and model.enabled
                and model.is_chat_selectable
            ):
                return model
        return None
