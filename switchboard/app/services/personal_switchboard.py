from __future__ import annotations

import json
import time
from datetime import datetime
from hashlib import sha256

from switchboard.app.models.api import ChatMessage
from switchboard.app.models.catalogue import ModelKind, ModelProfile, QualityTier
from switchboard.app.models.internal import (
    ClassificationResult,
    Complexity,
    NormalizedRequest,
    RoutingMode,
    Sensitivity,
    TaskType,
)
from switchboard.app.models.personal import (
    CompressionResult,
    FeedbackCreate,
    FeedbackRead,
    PersonalAskResponse,
    PersonalMemoryCreate,
    PersonalMemoryRead,
    PersonalModelRead,
    PersonalPromptRequest,
    PersonalRouteResponse,
    PremiumPrompt,
)
from switchboard.app.models.telemetry import (
    FeedbackRecord,
    MemoryItem,
    PersonalTelemetryRead,
    PersonalTelemetryRecord,
)
from switchboard.app.services.answer_quality import AnswerQualityHeuristic
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.context_compression import ContextCompressionService
from switchboard.app.services.local_runtime import (
    OllamaRuntimeService,
    RuntimeCommandResult,
)
from switchboard.app.utils.ids import new_request_id
from switchboard.app.utils.time import utc_now

LOCAL_KINDS = {ModelKind.MOCK, ModelKind.LOCAL, ModelKind.OPENAI_COMPATIBLE_LOCAL}
ROUTING_CACHE_VERSION = "personal-routing-v4-hot-model-runtime"


class PersonalRoutingError(ValueError):
    pass


class PersonalSwitchboardService:
    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        self.compressor = ContextCompressionService(container.cost_estimator)
        self.quality = AnswerQualityHeuristic()
        self.runtime = OllamaRuntimeService(container.personal_config)

    def models(self) -> list[PersonalModelRead]:
        return [
            PersonalModelRead.from_profile(
                model,
                self.container.personal_config.provider_enabled(model.provider),
            )
            for model in self.container.catalogue.models
        ]

    def route(self, request: PersonalPromptRequest, record: bool = True) -> PersonalRouteResponse:
        cache_key = self._cache_key(request)
        cache_allowed = (
            request.use_cache
            and not request.force_model
            and not request.baseline_model
            and not request.original_request_id
            and self.container.personal_config.preferences.cache_routing
        )
        if cache_allowed:
            cached = self.container.personal_telemetry_repository.get_cache(cache_key)
            if cached is not None:
                cached_response = PersonalRouteResponse.model_validate_json(cached.route_json)
                cached_response.request_id = new_request_id(
                    self.container.settings.request_id_prefix
                )
                cached_response.cache_hit = True
                cached_response.compression = self.compressor.compress(
                    request.prompt,
                    task_type=TaskType(cached_response.task_type),
                    sensitivity=Sensitivity(cached_response.sensitivity),
                )
                model = self.container.catalogue.get(cached_response.recommended_model)
                if model is not None and model.kind == ModelKind.MANUAL_SUBSCRIPTION:
                    cached_response.premium_prompt = self._premium_prompt(
                        model,
                        request.prompt,
                        cached_response.compression,
                    )
                if "CACHE_HIT" not in cached_response.reason_codes:
                    cached_response.reason_codes.append("CACHE_HIT")
                if record:
                    self._record(cached_response, status="recommended_cache_hit")
                return cached_response

        normalized = self._normalize(request)
        classification = self.container.classifier.classify(normalized)
        compression = self.compressor.compress(
            request.prompt,
            task_type=classification.task_type,
            sensitivity=classification.sensitivity,
        )
        loaded_models = sorted(self.runtime.list_loaded_models())
        router_selected, reason_codes = self._select_model(classification, loaded_models)
        feedback_selected = self._apply_feedback_preference(
            request,
            normalized.workflow_id,
            classification,
            router_selected,
            reason_codes,
        )
        try:
            selected = self._apply_force_model(
                request,
                classification,
                feedback_selected,
                reason_codes,
            )
        except PersonalRoutingError:
            if record and request.force_model:
                self._record_override_blocked(
                    request,
                    normalized,
                    classification,
                    compression,
                    router_selected,
                    reason_codes,
                )
            raise
        self._append_final_model_reason_codes(selected, classification, loaded_models, reason_codes)
        next_best = self._next_best_alternative(selected)
        output_tokens = normalized.max_tokens or 256
        cost = self.container.cost_estimator.estimate(
            selected,
            compression.compressed_estimated_tokens,
            output_tokens,
        )
        requires_confirmation = self._requires_confirmation(selected)
        baseline_model = self._baseline_model(request)
        baseline_route_kind = self._baseline_route_kind(baseline_model)
        baseline_source = "user_supplied" if request.baseline_model else "config_default"
        premium_spent, premium_saved = self._premium_units(selected, request, baseline_route_kind)
        api_cost_saved = self._api_cost_saved(
            selected,
            baseline_model,
            compression.compressed_estimated_tokens,
            output_tokens,
            cost.total_cost_usd,
        )

        response = PersonalRouteResponse(
            request_id=normalized.request_id,
            user_id=normalized.tenant_id,
            project=normalized.workflow_id,
            mode=normalized.routing_mode.value,
            task_type=classification.task_type.value,
            complexity=classification.complexity.value,
            sensitivity=classification.sensitivity.value,
            selected_model=selected.model_id,
            selected_provider=selected.provider,
            recommended_model=selected.model_id,
            recommended_provider=selected.provider,
            route_kind=selected.kind.value,
            scarce_model=selected.scarce,
            requires_confirmation=requires_confirmation,
            estimated_input_tokens=compression.compressed_estimated_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=cost.total_cost_usd,
            estimated_premium_units=premium_spent,
            estimated_premium_units_saved=premium_saved,
            compression=compression,
            confidence=classification.confidence,
            uncertainty_reasons=classification.uncertainty_reasons,
            privacy_note=self._privacy_note(selected, classification),
            next_best_alternative=next_best.model_id if next_best else None,
            premium_prompt=self._premium_prompt(selected, request.prompt, compression)
            if selected.kind == ModelKind.MANUAL_SUBSCRIPTION
            else None,
            explanation=self._explanation(selected, requires_confirmation),
            reason_codes=[
                *classification.reason_codes,
                *reason_codes,
            ],
            router_selected_model=router_selected.model_id,
            user_forced_model=request.force_model,
            final_selected_model=selected.model_id,
            override_used=bool(request.force_model),
            override_reason=request.override_reason,
            escalation_used=request.escalation_used,
            original_request_id=request.original_request_id,
            baseline_model=baseline_model.model_id if baseline_model else request.baseline_model,
            baseline_route_kind=baseline_route_kind,
            baseline_source=baseline_source,
            premium_unit_spent=premium_spent,
            premium_unit_saved=premium_saved,
            estimated_api_cost_saved=api_cost_saved,
            performance_mode=self.container.personal_config.local_runtime.performance_mode,
            selected_model_loaded=selected.model_id in loaded_models,
            model_switch_avoided="MODEL_SWITCH_AVOIDED" in reason_codes,
            cold_start_expected=(
                selected.provider == "ollama"
                and selected.kind in LOCAL_KINDS
                and selected.model_id not in loaded_models
            ),
            loaded_local_models=loaded_models,
        )
        if requires_confirmation:
            response.reason_codes.append("SCARCE_MODEL_REQUIRES_CONFIRMATION")
        if selected.kind in LOCAL_KINDS:
            response.reason_codes.append("LOCAL_MODEL_PREFERRED")
        if (
            record
            and cache_allowed
        ):
            cache_response = response.model_copy(deep=True)
            cache_response.compression.compressed_prompt = None
            cache_response.premium_prompt = None
            self.container.personal_telemetry_repository.set_cache(
                cache_key,
                response.project,
                response.mode,
                cache_response.model_dump_json(),
            )
        if record:
            self._record(response, status="recommended")
        return response

    async def ask(self, request: PersonalPromptRequest) -> PersonalAskResponse:
        route = self.route(request, record=False)
        if route.requires_confirmation or route.route_kind == ModelKind.MANUAL_SUBSCRIPTION.value:
            route.called_model = False
            route.recommended_only = True
            self._record(route, status="requires_confirmation")
            return PersonalAskResponse(
                request_id=route.request_id,
                recommendation=route,
                suggested_compressed_prompt=route.compression.compressed_prompt,
                status="requires_confirmation",
            )

        model = self.container.catalogue.get(route.recommended_model)
        adapter = self.container.providers.get(route.recommended_provider)
        if model is None or adapter is None:
            route.reason_codes.append("PERSONAL_PROVIDER_NOT_CONFIGURED")
            self._record(route, status="provider_unavailable")
            return PersonalAskResponse(
                request_id=route.request_id,
                recommendation=route,
                suggested_compressed_prompt=route.compression.compressed_prompt,
                status="provider_unavailable",
            )

        normalized = self._normalize(request, request_id=route.request_id)
        classification = self.container.classifier.classify(normalized)
        provider_request = self._provider_request(normalized, request, classification)
        if provider_request is not normalized:
            route.reason_codes.append("SOURCE_GROUNDED_SUMMARY_PROMPT")
        try:
            provider_response = await adapter.complete_chat(provider_request, model)
        except RuntimeError as exc:
            if route.recommended_provider != "ollama":
                route.reason_codes.append("PERSONAL_PROVIDER_UNAVAILABLE")
                self._record(route, status="provider_unavailable")
                return PersonalAskResponse(
                    request_id=route.request_id,
                    recommendation=route,
                    suggested_compressed_prompt=route.compression.compressed_prompt,
                    status="provider_unavailable",
                )
            fallback = self._mock_fallback_model(classification)
            fallback_adapter = self.container.providers.get("mock")
            if fallback is None or fallback_adapter is None:
                route.reason_codes.append("PERSONAL_PROVIDER_UNAVAILABLE")
                self._record(route, status="provider_unavailable")
                return PersonalAskResponse(
                    request_id=route.request_id,
                    recommendation=route,
                    suggested_compressed_prompt=route.compression.compressed_prompt,
                    quality_warning=True,
                    quality_notes=[f"Ollama unavailable and no mock fallback exists: {exc}"],
                    suggested_next_step="Start Ollama or route to a manual premium tool.",
                    status="provider_unavailable",
                )
            provider_response = await fallback_adapter.complete_chat(provider_request, fallback)
            provider_response.content = (
                "Fell back to mock because Ollama was unavailable.\n\n"
                f"{provider_response.content}"
            )
            self._apply_runtime_fallback(route, fallback)
        route.called_model = True
        route.recommended_only = False
        quality_warning, quality_notes, suggested_next_step = self.quality.assess(
            provider_response.content,
            request.prompt,
            classification,
            route,
            model.quality_tier,
        )
        self._record(route, status="called")
        return PersonalAskResponse(
            request_id=route.request_id,
            answer=provider_response.content,
            recommendation=route,
            quality_warning=quality_warning,
            quality_notes=quality_notes,
            suggested_next_step=suggested_next_step,
            status="called",
        )

    def usage(self) -> dict[str, object]:
        return self.container.personal_telemetry_repository.summary()

    def savings(
        self,
        days: int | None = 7,
        since: datetime | None = None,
    ) -> dict[str, object]:
        return self.container.personal_telemetry_repository.savings(days=days, since=since)

    def loaded_models(self) -> dict[str, object]:
        loaded = sorted(self.runtime.list_loaded_models())
        return {
            "ollama_enabled": self.runtime.enabled,
            "performance_mode": self.container.personal_config.local_runtime.performance_mode,
            "loaded_models": loaded,
        }

    def warm_model(
        self,
        model_id: str,
        allow_embedding: bool = False,
    ) -> RuntimeCommandResult:
        model = self.container.catalogue.get(model_id)
        if model is None:
            return RuntimeCommandResult(False, f"Unknown model: {model_id}", model_id)
        if model.provider != "ollama":
            return RuntimeCommandResult(
                False,
                f"Model is not an Ollama model: {model_id}",
                model_id,
            )
        if not model.enabled or not self.container.personal_config.provider_enabled("ollama"):
            return RuntimeCommandResult(
                False,
                f"Ollama model is disabled or unavailable: {model_id}",
                model_id,
            )
        if not allow_embedding and not model.is_chat_selectable:
            return RuntimeCommandResult(
                False,
                f"Refusing to warm embedding-only model for chat: {model_id}",
                model_id,
            )
        return self.runtime.warm_model(model.provider_model_name or model.model_id)

    def unload_model(self, model_id: str) -> RuntimeCommandResult:
        model = self.container.catalogue.get(model_id)
        if model is not None and model.provider == "ollama":
            return self.runtime.unload_model(model.provider_model_name or model.model_id)
        return self.runtime.unload_model(model_id)

    async def bench_models(self) -> list[dict[str, object]]:
        prompts = {
            "summary": "Summarise this sentence in five words: local models preserve quota.",
            "rewrite": "Rewrite this sentence politely: send me the document.",
            "coding": "Fix this Python snippet: print(customer_id)",
            "reasoning": "Compare speed and quality in one sentence.",
        }
        results: list[dict[str, object]] = []
        installed_ollama_models = self.runtime.list_installed_models()
        for model in self._available_models():
            if model.kind not in LOCAL_KINDS:
                continue
            adapter = self.container.providers.get(model.provider)
            prompt = prompts["coding"] if "coding" in model.good_for else prompts["summary"]
            normalized = self._normalize(PersonalPromptRequest(prompt=prompt, use_cache=False))
            started = time.perf_counter()
            error = None
            non_empty = False
            try:
                if adapter is None:
                    raise RuntimeError(f"Provider not configured: {model.provider}")
                response = await adapter.complete_chat(normalized, model)
                non_empty = bool(response.content.strip())
            except Exception as exc:  # pragma: no cover - provider availability is environmental
                error = str(exc)
            latency_ms = int((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "model": model.model_id,
                    "provider": model.provider,
                    "reachable": error is None,
                    "latency_ms": latency_ms,
                    "non_empty_response": non_empty,
                    "error": error,
                    "install_command": self._install_command(model)
                    if (
                        error
                        and model.provider == "ollama"
                        and model.model_id not in installed_ollama_models
                    )
                    else None,
                }
            )
            if (
                model.provider == "ollama"
                and self.container.personal_config.local_runtime.unload_after_benchmark
            ):
                unload = self.runtime.unload_model(model.model_id)
                results[-1]["unloaded_after_benchmark"] = unload.ok
                results[-1]["unload_message"] = unload.message
        return results

    def history(self, limit: int = 100) -> list[PersonalTelemetryRead]:
        return self.container.personal_telemetry_repository.list(limit=limit)

    def add_memory(self, item: PersonalMemoryCreate) -> PersonalMemoryRead:
        project = item.project or self.container.personal_config.profile.default_project
        return self.container.memory_repository.add(
            MemoryItem(
                project=project,
                title=item.title,
                content=item.content,
                tags_json=json.dumps(item.tags),
            )
        )

    def search_memory(self, query: str, project: str | None = None) -> list[PersonalMemoryRead]:
        resolved_project = project or self.container.personal_config.profile.default_project
        return self.container.memory_repository.search(resolved_project, query)

    def add_feedback(self, feedback: FeedbackCreate) -> FeedbackRead:
        return self.container.personal_telemetry_repository.add_feedback(
            FeedbackRecord(
                request_id=feedback.request_id,
                rating=feedback.rating,
                note=feedback.note,
                preferred_model=feedback.preferred_model,
            )
        )

    def _apply_runtime_fallback(
        self,
        route: PersonalRouteResponse,
        fallback: ModelProfile,
    ) -> None:
        route.selected_model = fallback.model_id
        route.selected_provider = fallback.provider
        route.recommended_model = fallback.model_id
        route.recommended_provider = fallback.provider
        route.route_kind = fallback.kind.value
        route.scarce_model = fallback.scarce
        route.requires_confirmation = False
        route.final_selected_model = fallback.model_id
        route.selected_model_loaded = False
        route.cold_start_expected = False
        route.privacy_note = "Fell back to mock because Ollama was unavailable."
        route.explanation = "Ollama was unavailable, so Switchboard used the mock fallback."
        route.reason_codes.append("MOCK_FALLBACK_USED")

    def _mock_fallback_model(
        self,
        classification: ClassificationResult,
    ) -> ModelProfile | None:
        mock_models = [
            model
            for model in self.container.catalogue.models
            if model.enabled
            and self.container.personal_config.provider_enabled(model.provider)
            and model.kind == ModelKind.MOCK
            and model.is_chat_selectable
        ]
        if not mock_models:
            return None
        if classification.task_type in {TaskType.CODING, TaskType.DEBUGGING}:
            return self._best_for(mock_models, "coding") or self._best_local(
                mock_models,
                QualityTier.MEDIUM,
            )
        if classification.complexity == Complexity.HIGH:
            return self._best_local(mock_models, QualityTier.FRONTIER)
        if classification.task_type in {
            TaskType.SUMMARISATION,
            TaskType.CLASSIFICATION,
            TaskType.EXTRACTION,
            TaskType.REWRITE,
        }:
            return self._best_local(mock_models, QualityTier.SMALL)
        return self._best_local(mock_models, QualityTier.MEDIUM)

    def _install_command(self, model: ModelProfile) -> str | None:
        if model.provider != "ollama":
            return None
        return f"ollama pull {model.provider_model_name or model.model_id.split('/', 1)[-1]}"

    def _normalize(
        self, request: PersonalPromptRequest, request_id: str | None = None
    ) -> NormalizedRequest:
        config = self.container.personal_config
        mode = request.mode or config.preferences.default_mode
        project = request.project or config.profile.default_project
        input_tokens = self.container.cost_estimator.estimate_text_tokens(request.prompt)
        return NormalizedRequest(
            request_id=request_id or new_request_id(self.container.settings.request_id_prefix),
            tenant_id=config.profile.user_id,
            application_id="personal-switchboard",
            workflow_id=project,
            environment=self.container.settings.environment,
            messages=[ChatMessage(role="user", content=request.prompt)],
            input_token_estimate=input_tokens,
            requested_model="personal/auto",
            metadata=request.metadata,
            routing_mode=RoutingMode.ACTIVE if mode == "auto" else RoutingMode(mode),
            max_tokens=256,
            created_at=utc_now(),
        )

    def _provider_request(
        self,
        normalized: NormalizedRequest,
        request: PersonalPromptRequest,
        classification: ClassificationResult,
    ) -> NormalizedRequest:
        if classification.task_type != TaskType.SUMMARISATION:
            return normalized
        source = self.quality.source_text_from_prompt(request.prompt)
        source_fact_count = (
            self.quality.distinct_fact_count(source)
            if source and len(source.split()) <= 30
            else None
        )
        system_prompt = self._summary_grounding_prompt(
            strict=request.strict,
            source_fact_count=source_fact_count,
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=request.prompt),
        ]
        input_tokens = self.container.cost_estimator.estimate_text_tokens(
            f"{system_prompt}\n\n{request.prompt}"
        )
        return normalized.model_copy(
            update={
                "messages": messages,
                "input_token_estimate": input_tokens,
            }
        )

    def _summary_grounding_prompt(
        self,
        strict: bool = False,
        source_fact_count: int | None = None,
    ) -> str:
        instructions = [
            "You are performing source-grounded summarisation.",
            "Summarise only the provided source text.",
            "Do not invent facts.",
            "Do not add assumptions.",
            "Do not add likely consequences unless explicitly stated in the source.",
            (
                "If the user asks for N bullets but the source contains fewer than N "
                "distinct facts, produce fewer bullets and say: \"Only X distinct facts "
                "were present in the source.\""
            ),
            (
                "Place that source-limitation note after the bullets, and make X match "
                "the number of factual bullets you produced."
            ),
            "Do not create filler bullets.",
            "Do not output blank bullets, placeholders, or bullets with no factual content.",
            "Do not create meta bullets unless needed as a final source-limitation note.",
            "Preserve the user's requested style where possible.",
        ]
        if source_fact_count is not None:
            instructions.append(
                "Switchboard estimates this short source contains about "
                f"{source_fact_count} distinct fact(s). Do not produce more factual "
                "bullets than that unless the source explicitly contains more."
            )
        if strict:
            instructions.extend(
                [
                    "Strict mode is enabled.",
                    "Every bullet must be directly supported by the source text.",
                    "When uncertain, omit the bullet instead of inferring.",
                ]
            )
        return "\n".join(f"- {instruction}" for instruction in instructions)

    def _available_models(self) -> list[ModelProfile]:
        return [
            model
            for model in self.container.catalogue.models
            if model.enabled and self.container.personal_config.provider_enabled(model.provider)
            and model.is_chat_selectable
        ]

    def _chat_excluded_models(self) -> list[ModelProfile]:
        return [
            model
            for model in self.container.catalogue.models
            if model.enabled
            and self.container.personal_config.provider_enabled(model.provider)
            and not model.is_chat_selectable
        ]

    def _select_model(
        self,
        classification: ClassificationResult,
        loaded_models: list[str] | None = None,
    ) -> tuple[ModelProfile, list[str]]:
        selected, reason_codes = self._select_ideal_model(classification)
        selected = self._apply_hot_model_preference(
            selected,
            classification,
            loaded_models or [],
            reason_codes,
        )
        return selected, reason_codes

    def _select_ideal_model(
        self,
        classification: ClassificationResult,
    ) -> tuple[ModelProfile, list[str]]:
        available = self._available_models()
        local_models = [model for model in available if model.kind in LOCAL_KINDS]
        manual_models = [
            model for model in available if model.kind == ModelKind.MANUAL_SUBSCRIPTION
        ]
        cloud_models = [model for model in available if model.kind == ModelKind.CLOUD_API]
        preferences = self.container.personal_config.preferences
        reason_codes: list[str] = ["PERSONAL_LOCAL_FIRST_ENABLED"]
        if self._chat_excluded_models():
            reason_codes.append("EMBEDDING_MODEL_SKIPPED_FOR_CHAT")
        runtime_mode = self.container.personal_config.local_runtime.performance_mode
        if runtime_mode == "memory_saver":
            reason_codes.append("MEMORY_SAVER_MODE_ACTIVE")
        elif runtime_mode == "low_latency":
            reason_codes.append("LOW_LATENCY_MODE_ACTIVE")
        else:
            reason_codes.append("BALANCED_RUNTIME_MODE_ACTIVE")

        sensitive = classification.sensitivity in {
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
            Sensitivity.PRIVATE_PERSONAL,
        }
        if preferences.private_mode and sensitive:
            reason_codes.extend(
                ["PERSONAL_PRIVATE_MODE_CLOUD_BLOCKED", "PRIVATE_MODE_ENABLED", "CLOUD_DISABLED"]
            )
            local_complex = self._prefer_model_ids(
                local_models,
                ["ollama/deepseek-r1:8b", "ollama/gemma3:12b"],
            )
            if classification.task_type in {
                TaskType.SUMMARISATION,
                TaskType.CLASSIFICATION,
                TaskType.EXTRACTION,
                TaskType.REWRITE,
            }:
                reason_codes.extend(
                    [
                        "PERSONAL_SENSITIVE_SIMPLE_TASK_KEPT_LOCAL",
                        "PERSONAL_SIMPLE_TASK_ROUTED_TO_FREE_LOCAL_MODEL",
                        "SENSITIVE_BUT_SIMPLE_TASK",
                        "SENSITIVITY_DOES_NOT_IMPLY_FRONTIER",
                    ]
                )
                if classification.sensitivity in {
                    Sensitivity.REGULATED,
                    Sensitivity.PRIVATE_PERSONAL,
                }:
                    return self._private_summary_model(local_models), reason_codes
                return self._simple_task_model(local_models), reason_codes
            if (
                classification.complexity == Complexity.HIGH
                or classification.task_type
                in {TaskType.REASONING, TaskType.PLANNING, TaskType.ARCHITECTURE_DESIGN}
            ):
                if local_complex is not None:
                    return local_complex, reason_codes
                return self._best_local(local_models, QualityTier.FRONTIER), reason_codes
            return self._best_local(local_models, QualityTier.MEDIUM), reason_codes

        if (
            classification.task_type in {TaskType.CODING, TaskType.DEBUGGING}
            and classification.complexity == Complexity.HIGH
        ):
            coding_local = self._complex_coding_model(local_models)
            if coding_local is not None and coding_local.kind != ModelKind.MOCK:
                reason_codes.extend(["PERSONAL_CODING_LOCAL_MODEL_PREFERRED", "CODING_TASK"])
                return coding_local, reason_codes
            codex = self._best_for(manual_models, "coding")
            if codex is not None and not sensitive:
                reason_codes.extend(
                    [
                        "PERSONAL_COMPLEX_CODING_PREMIUM_RECOMMENDED",
                        "MANUAL_PREMIUM_RECOMMENDED",
                        "LOW_CONFIDENCE_ESCALATION"
                        if classification.confidence < 0.55
                        else "COMPLEX_REASONING_REQUEST",
                    ]
                )
                return codex, reason_codes

        if (
            classification.complexity == Complexity.HIGH
            or classification.task_type
            in {TaskType.REASONING, TaskType.PLANNING, TaskType.ARCHITECTURE_DESIGN}
        ) and not sensitive:
            local_reasoning = self._complex_reasoning_model(local_models)
            if local_reasoning is not None and local_reasoning.kind != ModelKind.MOCK:
                reason_codes.append("PERSONAL_COMPLEX_REASONING_LOCAL_MODEL_PREFERRED")
                return local_reasoning, reason_codes
            if not preferences.allow_cloud:
                premium = self._best_for(manual_models, "reasoning") or self._best_for(
                    manual_models, "planning"
                )
                if premium is not None:
                    reason_codes.extend(
                        [
                            "PERSONAL_CLOUD_DISABLED_PREMIUM_RECOMMENDATION_ONLY",
                            "PERSONAL_SCARCE_MODEL_NOT_CALLED_AUTOMATICALLY",
                            "MANUAL_PREMIUM_RECOMMENDED",
                            "CLOUD_DISABLED",
                        ]
                    )
                    return premium, reason_codes
            if preferences.allow_cloud and cloud_models:
                reason_codes.append("PERSONAL_CLOUD_ALLOWED_BY_USER")
                return self._best_tier(cloud_models, QualityTier.FRONTIER), reason_codes

        if classification.task_type in {
            TaskType.SUMMARISATION,
            TaskType.CLASSIFICATION,
            TaskType.EXTRACTION,
            TaskType.REWRITE,
        }:
            reason_codes.append("PERSONAL_SIMPLE_TASK_ROUTED_TO_FREE_LOCAL_MODEL")
            return self._simple_task_model(local_models), reason_codes

        if classification.task_type in {TaskType.CODING, TaskType.DEBUGGING}:
            coding_local = self._coding_model(local_models)
            if coding_local is not None:
                reason_codes.extend(["PERSONAL_CODING_LOCAL_MODEL_PREFERRED", "CODING_TASK"])
                return coding_local, reason_codes

        if classification.confidence < 0.55:
            reason_codes.extend(
                [
                    "LOW_CONFIDENCE_SAFE_LOCAL_ROUTE",
                    "PERSONAL_AMBIGUOUS_REQUEST_KEPT_LOCAL",
                ]
            )
            if "PROMPT_INJECTION_ATTEMPT" in classification.reason_codes:
                reason_codes.append("PERSONAL_PROMPT_INJECTION_KEPT_LOCAL")
            return self._best_local(local_models, QualityTier.MEDIUM), reason_codes

        reason_codes.append("PERSONAL_DEFAULT_MEDIUM_LOCAL_ROUTE")
        return self._general_reasoning_model(local_models), reason_codes

    def _apply_hot_model_preference(
        self,
        selected: ModelProfile,
        classification: ClassificationResult,
        loaded_model_ids: list[str],
        reason_codes: list[str],
    ) -> ModelProfile:
        loaded = set(loaded_model_ids)
        if selected.provider == "ollama":
            if selected.model_id in loaded:
                reason_codes.append("OLLAMA_MODEL_ALREADY_LOADED")
                return selected
            reason_codes.append("OLLAMA_MODEL_NOT_LOADED")

        runtime = self.container.personal_config.local_runtime
        if not runtime.reuse_hot_model_if_good_enough or not loaded:
            return selected

        hot_candidates = [
            model
            for model in self._available_models()
            if model.model_id in loaded
            and model.provider == "ollama"
            and model.kind in LOCAL_KINDS
            and model.is_chat_selectable
        ]
        for candidate in self._sort_models(hot_candidates):
            if self._hot_model_good_enough(candidate, selected, classification):
                if candidate.model_id != selected.model_id:
                    if "OLLAMA_MODEL_NOT_LOADED" in reason_codes:
                        reason_codes.remove("OLLAMA_MODEL_NOT_LOADED")
                    reason_codes.extend(
                        [
                            "OLLAMA_MODEL_ALREADY_LOADED",
                            "HOT_MODEL_REUSED",
                            "HOT_MODEL_GOOD_ENOUGH",
                            "MODEL_SWITCH_AVOIDED",
                        ]
                    )
                return candidate

        if (
            selected.provider == "ollama"
            and selected.model_id not in loaded
            and (
                classification.task_type in {TaskType.CODING, TaskType.DEBUGGING}
                or classification.complexity == Complexity.HIGH
            )
        ):
            reason_codes.append("SPECIALIST_MODEL_SWITCH_JUSTIFIED")
        return selected

    def _hot_model_good_enough(
        self,
        candidate: ModelProfile,
        selected: ModelProfile,
        classification: ClassificationResult,
    ) -> bool:
        if candidate.model_id == selected.model_id:
            return True
        runtime = self.container.personal_config.local_runtime
        if classification.task_type in {TaskType.CODING, TaskType.DEBUGGING}:
            return "coding" in candidate.good_for or "debugging" in candidate.good_for
        if classification.complexity == Complexity.HIGH:
            return (
                candidate.quality_tier == QualityTier.FRONTIER
                and bool(
                    {
                        "reasoning",
                        "planning",
                        "debugging",
                        "complex_debugging",
                        "tradeoff_analysis",
                        "architecture_design",
                    }
                    & set(candidate.good_for)
                )
            )
        if classification.task_type in {
            TaskType.SUMMARISATION,
            TaskType.CLASSIFICATION,
            TaskType.EXTRACTION,
            TaskType.REWRITE,
        }:
            if not runtime.prefer_hot_model_for_simple_tasks:
                return False
            return bool(
                {
                    "summarisation",
                    "classification",
                    "extraction",
                    "rewrite",
                    "reasoning",
                    "general",
                    "simple_qa",
                }
                & set(candidate.good_for)
            )
        if runtime.performance_mode == "memory_saver":
            return bool({"reasoning", "planning", "summarisation"} & set(candidate.good_for))
        return candidate.quality_tier in {QualityTier.MEDIUM, QualityTier.FRONTIER}

    def _append_final_model_reason_codes(
        self,
        selected: ModelProfile,
        classification: ClassificationResult,
        loaded_models: list[str],
        reason_codes: list[str],
    ) -> None:
        del classification

        def add(code: str) -> None:
            if code not in reason_codes:
                reason_codes.append(code)

        if selected.provider != "ollama":
            return
        add("LOCAL_OLLAMA_MODEL_SELECTED")
        if selected.model_id == "ollama/llama3.2:3b":
            add("OLLAMA_FAST_MODEL_SELECTED")
        elif selected.model_id == "ollama/qwen3:8b":
            add("OLLAMA_GENERAL_MODEL_SELECTED")
        elif selected.model_id == "ollama/qwen2.5-coder:7b":
            add("OLLAMA_CODING_MODEL_SELECTED")
        elif selected.model_id == "ollama/deepseek-r1:8b":
            add("OLLAMA_REASONING_MODEL_SELECTED")
        if selected.model_id not in set(loaded_models):
            add("COLD_START_POSSIBLE")

    def _apply_force_model(
        self,
        request: PersonalPromptRequest,
        classification: ClassificationResult,
        router_selected: ModelProfile,
        reason_codes: list[str],
    ) -> ModelProfile:
        if not request.force_model:
            return router_selected
        forced = self.container.catalogue.get(request.force_model)
        if forced is None:
            raise PersonalRoutingError(f"Unknown force-model: {request.force_model}")
        provider_enabled = self.container.personal_config.provider_enabled(forced.provider)
        if not forced.enabled or not provider_enabled:
            raise PersonalRoutingError(f"Force-model is disabled or unavailable: {forced.model_id}")
        if not forced.is_chat_selectable:
            raise PersonalRoutingError(
                f"Force-model is not valid for chat responses: {forced.model_id}"
            )
        sensitive = classification.sensitivity in {
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
            Sensitivity.PRIVATE_PERSONAL,
        }
        if (
            self.container.personal_config.preferences.private_mode
            and sensitive
            and forced.kind in {ModelKind.CLOUD_API, ModelKind.MANUAL_SUBSCRIPTION}
        ):
            raise PersonalRoutingError(
                f"Force-model {forced.model_id} is blocked by private mode "
                "for sensitive content."
            )
        if (
            forced.kind == ModelKind.CLOUD_API
            and not request.allow_cloud_once
            and not self.container.personal_config.preferences.allow_cloud
        ):
            raise PersonalRoutingError(
                f"Cloud force-model {forced.model_id} blocked because allow_cloud=false. "
                "Use --allow-cloud-once only for non-sensitive prompts."
            )
        reason_codes.extend(["USER_FORCE_MODEL_REQUESTED", "USER_FORCE_MODEL_APPLIED"])
        if forced.kind == ModelKind.CLOUD_API and request.allow_cloud_once:
            reason_codes.append("USER_ALLOW_CLOUD_ONCE")
        if forced.kind == ModelKind.MANUAL_SUBSCRIPTION:
            reason_codes.extend(
                ["MANUAL_PREMIUM_RECOMMENDED", "PERSONAL_SCARCE_MODEL_NOT_CALLED_AUTOMATICALLY"]
            )
        return forced

    def _apply_feedback_preference(
        self,
        request: PersonalPromptRequest,
        project: str,
        classification: ClassificationResult,
        selected: ModelProfile,
        reason_codes: list[str],
    ) -> ModelProfile:
        if request.force_model:
            return selected
        preferred_model_id = (
            self.container.personal_telemetry_repository.preferred_model_from_feedback(
                project=project,
                task_type=classification.task_type.value,
                current_model=selected.model_id,
            )
        )
        if not preferred_model_id:
            return selected
        preferred = self.container.catalogue.get(preferred_model_id)
        if preferred is None:
            return selected
        if not preferred.enabled or not self.container.personal_config.provider_enabled(
            preferred.provider
        ):
            return selected
        if not preferred.is_chat_selectable or preferred.kind not in LOCAL_KINDS:
            return selected
        if self._quality_rank(preferred) < self._quality_rank(selected):
            return selected
        if preferred.model_id == selected.model_id:
            return selected
        reason_codes.append("PREVIOUS_FEEDBACK_CONSIDERED")
        return preferred

    def _quality_rank(self, model: ModelProfile) -> int:
        ranks = {
            QualityTier.SMALL: 1,
            QualityTier.MEDIUM: 2,
            QualityTier.FRONTIER: 3,
            QualityTier.EMBEDDING: 0,
        }
        return ranks.get(model.quality_tier, 0)

    def _best_local(self, models: list[ModelProfile], tier: QualityTier) -> ModelProfile:
        matching = [model for model in models if model.quality_tier == tier]
        if matching:
            return self._sort_models(matching)[0]
        if models:
            return self._sort_models(models)[0]
        raise ValueError("No local or mock models are enabled in personal config")

    def _best_for(self, models: list[ModelProfile], capability: str) -> ModelProfile | None:
        matching = [model for model in models if capability in model.good_for]
        return self._sort_models(matching)[0] if matching else None

    def _prefer_model_ids(
        self,
        models: list[ModelProfile],
        model_ids: list[str],
    ) -> ModelProfile | None:
        by_id = {model.model_id: model for model in models}
        for model_id in model_ids:
            if model_id in by_id:
                return by_id[model_id]
        return None

    def _simple_task_model(self, models: list[ModelProfile]) -> ModelProfile:
        return (
            self._prefer_model_ids(models, ["ollama/llama3.2:3b", "ollama/qwen3:8b"])
            or self._best_local(models, QualityTier.SMALL)
        )

    def _private_summary_model(self, models: list[ModelProfile]) -> ModelProfile:
        return (
            self._prefer_model_ids(
                models,
                ["ollama/qwen3:8b", "ollama/llama3.2:3b", "ollama/gemma3:12b"],
            )
            or self._best_local(models, QualityTier.MEDIUM)
        )

    def _general_reasoning_model(self, models: list[ModelProfile]) -> ModelProfile:
        return (
            self._prefer_model_ids(models, ["ollama/qwen3:8b", "ollama/gemma3:12b"])
            or self._best_local(models, QualityTier.MEDIUM)
        )

    def _coding_model(self, models: list[ModelProfile]) -> ModelProfile | None:
        return (
            self._prefer_model_ids(models, ["ollama/qwen2.5-coder:7b", "ollama/qwen3:8b"])
            or self._best_for(models, "coding")
        )

    def _complex_reasoning_model(self, models: list[ModelProfile]) -> ModelProfile | None:
        return self._prefer_model_ids(
            models,
            ["ollama/deepseek-r1:8b", "ollama/gemma3:12b", "ollama/qwen3:8b"],
        )

    def _complex_coding_model(self, models: list[ModelProfile]) -> ModelProfile | None:
        return self._prefer_model_ids(
            models,
            ["ollama/qwen2.5-coder:7b", "ollama/deepseek-r1:8b", "ollama/gemma3:12b"],
        ) or self._best_for(models, "coding")

    def _best_tier(self, models: list[ModelProfile], tier: QualityTier) -> ModelProfile:
        matching = [model for model in models if model.quality_tier == tier]
        return self._sort_models(matching or models)[0]

    def _sort_models(self, models: list[ModelProfile]) -> list[ModelProfile]:
        return sorted(
            models,
            key=lambda model: (
                model.scarce,
                model.input_cost_per_million_tokens + model.output_cost_per_million_tokens,
                model.average_latency_ms,
                model.model_id,
            ),
        )

    def _requires_confirmation(self, model: ModelProfile) -> bool:
        return model.kind == ModelKind.MANUAL_SUBSCRIPTION or (
            model.scarce
            and self.container.personal_config.preferences.require_confirmation_for_scarce_models
        )

    def _explanation(self, model: ModelProfile, requires_confirmation: bool) -> str:
        if model.kind == ModelKind.MANUAL_SUBSCRIPTION:
            return (
                f"Recommendation only: use {model.display_name} manually if this task is worth "
                "scarce premium usage. Switchboard will not automate subscription web UIs."
            )
        if requires_confirmation:
            return f"{model.display_name} is scarce, so confirmation is required before calling it."
        return (
            f"Selected {model.display_name} because it fits the task and current "
            "local-first preferences."
        )

    def _next_best_alternative(self, selected: ModelProfile) -> ModelProfile | None:
        available = [
            model for model in self._available_models() if model.model_id != selected.model_id
        ]
        if not available:
            return None
        same_capability = [
            model
            for model in available
            if any(capability in model.good_for for capability in selected.good_for)
        ]
        return self._sort_models(same_capability or available)[0]

    def _privacy_note(self, model: ModelProfile, classification: ClassificationResult) -> str:
        if classification.sensitivity in {
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
            Sensitivity.PRIVATE_PERSONAL,
        }:
            if model.kind in LOCAL_KINDS:
                return "Sensitive/private content is kept on a local or mock route by default."
            return (
                "Sensitive/private content should not be sent to this route without "
                "explicit review."
            )
        if model.kind == ModelKind.CLOUD_API:
            return "Cloud route is only allowed because user preferences enable cloud API use."
        if model.kind == ModelKind.MANUAL_SUBSCRIPTION:
            return "Manual subscription recommendation only; no web UI automation is performed."
        return "No prompt or response body is stored by default."

    def _premium_prompt(
        self,
        model: ModelProfile,
        prompt: str,
        compression: CompressionResult,
    ) -> PremiumPrompt:
        compressed_prompt = (
            compression.compressed_prompt
            if hasattr(compression, "compressed_prompt") and compression.compressed_prompt
            else prompt
        )
        return PremiumPrompt(
            title="Ready-to-paste premium prompt",
            recommended_tool=model.display_name,
            ready_to_paste_prompt=(
                "Please help with the task below. Be precise, flag uncertainty, and preserve "
                "privacy-sensitive details.\n\n"
                f"{compressed_prompt}"
            ),
            why_this_tool=f"{model.display_name} is recommended for the complexity of this task.",
            what_to_try_locally_first=(
                "Try a local/mock medium model first for a draft, then use this premium prompt "
                "only if the local answer is too weak."
            ),
            estimated_tokens_saved=getattr(compression, "estimated_tokens_saved", 0),
        )

    def _record_override_blocked(
        self,
        request: PersonalPromptRequest,
        normalized: NormalizedRequest,
        classification: ClassificationResult,
        compression: object,
        router_selected: ModelProfile,
        reason_codes: list[str],
    ) -> None:
        output_tokens = normalized.max_tokens or 256
        cost = self.container.cost_estimator.estimate(
            router_selected,
            getattr(compression, "compressed_estimated_tokens", normalized.input_token_estimate),
            output_tokens,
        )
        baseline_model = self._baseline_model(request)
        response = PersonalRouteResponse(
            request_id=normalized.request_id,
            user_id=normalized.tenant_id,
            project=normalized.workflow_id,
            mode=normalized.routing_mode.value,
            task_type=classification.task_type.value,
            complexity=classification.complexity.value,
            sensitivity=classification.sensitivity.value,
            selected_model=router_selected.model_id,
            selected_provider=router_selected.provider,
            recommended_model=router_selected.model_id,
            recommended_provider=router_selected.provider,
            route_kind=router_selected.kind.value,
            scarce_model=router_selected.scarce,
            requires_confirmation=False,
            estimated_input_tokens=getattr(
                compression,
                "compressed_estimated_tokens",
                normalized.input_token_estimate,
            ),
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=cost.total_cost_usd,
            estimated_premium_units=0.0,
            estimated_premium_units_saved=0.0,
            compression=compression,
            confidence=classification.confidence,
            uncertainty_reasons=classification.uncertainty_reasons,
            privacy_note="Override was blocked by safety settings.",
            next_best_alternative=None,
            explanation="Requested override was blocked by privacy or provider safety settings.",
            reason_codes=[
                *classification.reason_codes,
                *reason_codes,
                "USER_FORCE_MODEL_REQUESTED",
                "OVERRIDE_SAFETY_BLOCKED",
            ],
            router_selected_model=router_selected.model_id,
            user_forced_model=request.force_model,
            final_selected_model=router_selected.model_id,
            override_used=bool(request.force_model),
            override_reason=request.override_reason,
            override_safety_blocked=True,
            escalation_used=request.escalation_used,
            original_request_id=request.original_request_id,
            baseline_model=baseline_model.model_id if baseline_model else request.baseline_model,
            baseline_route_kind=self._baseline_route_kind(baseline_model),
            baseline_source="user_supplied" if request.baseline_model else "config_default",
        )
        self._record(response, status="override_blocked")

    def _baseline_model(self, request: PersonalPromptRequest) -> ModelProfile | None:
        model_id = (
            request.baseline_model
            or self.container.personal_config.savings.default_baseline_model
        )
        return self.container.catalogue.get(model_id)

    def _baseline_route_kind(self, baseline: ModelProfile | None) -> str | None:
        return baseline.kind.value if baseline else None

    def _premium_units(
        self,
        selected: ModelProfile,
        request: PersonalPromptRequest,
        baseline_route_kind: str | None,
    ) -> tuple[float, float]:
        if selected.kind == ModelKind.MANUAL_SUBSCRIPTION:
            if request.force_model or request.escalation_used:
                return 1.0, 0.0
            return 0.0, 0.0
        if selected.kind in LOCAL_KINDS and baseline_route_kind in {
            ModelKind.MANUAL_SUBSCRIPTION.value,
            ModelKind.CLOUD_API.value,
        }:
            return 0.0, 1.0
        return 0.0, 0.0

    def _api_cost_saved(
        self,
        selected: ModelProfile,
        baseline: ModelProfile | None,
        input_tokens: int,
        output_tokens: int,
        selected_cost: float,
    ) -> float:
        if baseline is None or baseline.kind != ModelKind.CLOUD_API:
            return 0.0
        baseline_cost = self.container.cost_estimator.estimate(
            baseline,
            input_tokens,
            output_tokens,
        ).total_cost_usd
        return max(0.0, round(baseline_cost - selected_cost, 8))

    def _cache_key(self, request: PersonalPromptRequest) -> str:
        config = self.container.personal_config
        preferences = config.preferences.model_dump(mode="json")
        model_fingerprint = ",".join(
            sorted(model.model_id for model in self.container.catalogue.models if model.enabled)
        )
        normalized_prompt = " ".join(request.prompt.lower().split())
        payload = {
            "cache_version": ROUTING_CACHE_VERSION,
            "prompt_hash": sha256(normalized_prompt.encode("utf-8")).hexdigest(),
            "mode": request.mode or config.preferences.default_mode,
            "project": request.project or config.profile.default_project,
            "preferences": preferences,
            "local_runtime": config.local_runtime.model_dump(mode="json"),
            "loaded_models": sorted(self.runtime.list_loaded_models()),
            "feedback": self.container.personal_telemetry_repository.feedback_summary(),
            "model_fingerprint": model_fingerprint,
        }
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _record(self, route: PersonalRouteResponse, status: str) -> None:
        self.container.personal_telemetry_repository.add(
            PersonalTelemetryRecord(
                request_id=route.request_id,
                user_id=route.user_id,
                project=route.project,
                mode=route.mode,
                task_type=route.task_type,
                complexity=route.complexity,
                sensitivity=route.sensitivity,
                selected_model=route.recommended_model,
                selected_provider=route.recommended_provider,
                route_kind=route.route_kind,
                scarce_model=route.scarce_model,
                required_confirmation=route.requires_confirmation,
                called_model=route.called_model,
                recommended_only=route.recommended_only,
                estimated_input_tokens=route.estimated_input_tokens,
                estimated_output_tokens=route.estimated_output_tokens,
                estimated_cost_usd=route.estimated_cost_usd,
                estimated_premium_units=route.estimated_premium_units,
                estimated_premium_units_saved=route.estimated_premium_units_saved,
                router_selected_model=route.router_selected_model or route.recommended_model,
                user_forced_model=route.user_forced_model,
                final_selected_model=route.final_selected_model or route.recommended_model,
                override_used=route.override_used,
                override_reason=route.override_reason,
                override_safety_blocked=route.override_safety_blocked,
                escalation_used=route.escalation_used,
                original_request_id=route.original_request_id,
                original_model=self._original_model(route.original_request_id),
                escalated_to_model=route.recommended_model if route.escalation_used else None,
                escalation_reason=route.override_reason if route.escalation_used else None,
                manual_recommendation=route.route_kind == ModelKind.MANUAL_SUBSCRIPTION.value,
                premium_unit_spent=route.premium_unit_spent,
                premium_unit_saved=route.premium_unit_saved,
                estimated_api_cost_saved=route.estimated_api_cost_saved,
                baseline_model=route.baseline_model,
                baseline_route_kind=route.baseline_route_kind,
                baseline_source=route.baseline_source,
                selected_model_loaded=route.selected_model_loaded,
                model_switch_avoided=route.model_switch_avoided,
                cold_start_expected=route.cold_start_expected,
                performance_mode=route.performance_mode,
                loaded_local_models_json=json.dumps(route.loaded_local_models),
                reason_codes_json=json.dumps(route.reason_codes),
                status=status,
                cache_hit=route.cache_hit,
            )
        )

    def _original_model(self, request_id: str | None) -> str | None:
        if not request_id:
            return None
        record = self.container.personal_telemetry_repository.get(request_id)
        if record is None:
            return None
        return record.final_selected_model or record.selected_model
