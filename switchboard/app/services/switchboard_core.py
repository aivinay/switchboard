from __future__ import annotations

import json
import re

from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.registry import BackendRegistry
from switchboard.app.models.api import ChatMessage
from switchboard.app.models.backends import (
    BackendCostType,
    BackendInfo,
    BackendRouteDecision,
    SwitchboardRequest,
    SwitchboardResponse,
    backend_display_name,
)
from switchboard.app.models.capabilities import (
    Capability,
    CapabilityDetection,
    RuntimeContext,
    ToolResult,
)
from switchboard.app.models.internal import NormalizedRequest, RoutingMode, Sensitivity
from switchboard.app.models.sessions import ChatMessageRead, ChatSessionRead
from switchboard.app.models.telemetry import BackendMetricRead, BackendMetricRecord
from switchboard.app.services.answer_confidence import (
    AnswerConfidenceResult,
    AnswerConfidenceService,
)
from switchboard.app.services.capabilities import CapabilityDetector
from switchboard.app.services.compression_layer import (
    CompressionLayer,
    NoCompressionLayer,
)
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.learned_router import LearnedRouter
from switchboard.app.services.llm_router import LlmRouter
from switchboard.app.services.quota import (
    PREMIUM_BACKENDS,
    QuotaLedgerService,
    QuotaWindowStatus,
)
from switchboard.app.services.response_sanitizer import ResponseSanitizer
from switchboard.app.services.runtime_context import RuntimeContextProvider
from switchboard.app.services.semantic_memory import SemanticMemoryService
from switchboard.app.services.sensitivity_escalator import (
    LearnedSensitivityEscalator,
)
from switchboard.app.services.session_context import ContextBuilder, SessionManager
from switchboard.app.services.tool_dispatcher import (
    VERIFIED_TOOL_CLASSES,
    LearnedToolDispatcher,
)
from switchboard.app.services.tools import ToolRegistry
from switchboard.app.storage.repositories import BackendMetricsRepository, ContextStore
from switchboard.app.utils.ids import new_request_id
from switchboard.app.utils.redaction import sanitize_provider_error


class SwitchboardCoreService:
    def __init__(
        self,
        *,
        registry: BackendRegistry,
        metrics: BackendMetricsRepository,
        container: ServiceContainer,
        compression: CompressionLayer | None = None,
        runtime_context_provider: RuntimeContextProvider | None = None,
        capability_detector: CapabilityDetector | None = None,
        tool_registry: ToolRegistry | None = None,
        context_store: ContextStore | None = None,
        session_manager: SessionManager | None = None,
        context_builder: ContextBuilder | None = None,
        response_sanitizer: ResponseSanitizer | None = None,
        router_mode: str = "rules",
        llm_router: LlmRouter | None = None,
        learned_router: LearnedRouter | None = None,
        semantic_memory: SemanticMemoryService | None = None,
        tool_dispatcher: LearnedToolDispatcher | None = None,
        sensitivity_escalator: LearnedSensitivityEscalator | None = None,
        answer_confidence: AnswerConfidenceService | None = None,
        quota_ledger: QuotaLedgerService | None = None,
    ) -> None:
        self.registry = registry
        self.metrics = metrics
        self.container = container
        self.compression = compression or NoCompressionLayer()
        self.runtime_context_provider = runtime_context_provider or RuntimeContextProvider()
        self.capability_detector = capability_detector or CapabilityDetector()
        self.tool_registry = tool_registry or ToolRegistry()
        self.context_store = context_store or container.context_store
        self.session_manager = session_manager or SessionManager(self.context_store)
        self.context_builder = context_builder or ContextBuilder()
        self.response_sanitizer = response_sanitizer or ResponseSanitizer()
        self.router_mode = (
            router_mode if router_mode in {"rules", "llm", "hybrid", "learned"} else "rules"
        )
        self.llm_router = llm_router
        self.learned_router = learned_router
        self.semantic_memory = semantic_memory
        self.tool_dispatcher = tool_dispatcher
        self.sensitivity_escalator = sensitivity_escalator
        self.answer_confidence = answer_confidence or AnswerConfidenceService()
        self.quota_ledger = quota_ledger or QuotaLedgerService(
            self.metrics,
            self.container.personal_config.quota,
        )

    def backends(self) -> list[BackendInfo]:
        return self.registry.list_backends()

    def metrics_list(self, limit: int = 20) -> list[BackendMetricRead]:
        return self.metrics.list(limit)

    def metrics_summary(self) -> dict[str, object]:
        return self.metrics.summary()

    def quota_status(self) -> dict[str, object]:
        return self.quota_ledger.snapshot()

    def preview_route(
        self,
        prompt: str,
        *,
        backend: str | None = None,
        project: str | None = None,
        model: str | None = None,
        timeout_s: int = 120,
        metadata: dict[str, object] | None = None,
    ) -> BackendRouteDecision:
        request = SwitchboardRequest(
            request_id=new_request_id(self.container.settings.request_id_prefix),
            prompt=prompt,
            project=project or self.container.personal_config.profile.default_project,
            model=model,
            timeout_s=timeout_s,
            private_mode=self.container.personal_config.preferences.private_mode,
            metadata={
                "context_message_count": 0,
                "context_summary_used": False,
                "context_recent_message_count": 0,
                "context_injected": False,
                "grounded_by_tool": False,
                "model_called": False,
                "followup_intent_reused": False,
                **dict(metadata or {}),
            },
        )
        runtime_context = self.runtime_context_provider.current()
        detection = self.capability_detector.detect(prompt)
        tool_result = self.tool_registry.resolve(
            prompt=prompt,
            detection=detection,
            context=runtime_context,
        )
        if tool_result is None:
            detection, tool_result = self._maybe_dispatch_learned_tool(
                prompt=prompt,
                detection=detection,
                context=runtime_context,
                request=request,
            )
        request.metadata.update(self._capability_metadata(detection))
        request.metadata.update(self._phase2_metadata(detection))
        if tool_result is not None and tool_result.success:
            request.metadata.update(self._tool_grounding_metadata(tool_result))
        elif tool_result is not None:
            request.metadata.update(self._tool_pass_through_metadata(tool_result))
        elif self._has_unconfigured_live_capability(detection):
            request.metadata.update(self._pass_through_metadata(detection))
        request = self.compression.compress(request)
        decision = self.route(request, forced_backend=backend)
        request.metadata.update(self._route_metadata(decision))
        return decision

    def ask(
        self,
        prompt: str,
        *,
        backend: str | None = None,
        project: str | None = None,
        model: str | None = None,
        timeout_s: int = 120,
        metadata: dict[str, object] | None = None,
        session_id: str | None = None,
        new_session: bool = False,
    ) -> SwitchboardResponse:
        session = self.session_manager.resolve_session(
            session_id=session_id,
            new_session=new_session,
        )
        request = SwitchboardRequest(
            request_id=new_request_id(self.container.settings.request_id_prefix),
            prompt=prompt,
            project=project or self.container.personal_config.profile.default_project,
            model=model,
            timeout_s=timeout_s,
            private_mode=self.container.personal_config.preferences.private_mode,
            metadata=dict(metadata or {}),
        )
        user_message = self.context_store.append_message(
            session_id=session.session_id,
            role="user",
            content=prompt,
            metadata={"request_id": request.request_id, "project": request.project},
        )
        request.metadata.update(
            {
                "session_id": session.session_id,
                "message_id": user_message.message_id,
                "user_message_id": user_message.message_id,
                "context_message_count": 0,
                "context_summary_used": False,
                "context_recent_message_count": 0,
                "context_injected": False,
                "grounded_by_tool": False,
                "model_called": False,
                "followup_intent_reused": False,
            }
        )
        runtime_context = self.runtime_context_provider.current()
        effective_prompt, reused_message_id = self._resolve_followup_prompt(
            prompt=prompt,
            session=session,
            current_message_id=user_message.message_id,
        )
        if reused_message_id:
            request.metadata.update(
                {
                    "followup_intent_reused": True,
                    "followup_source_message_id": reused_message_id,
                }
            )
        detection = self.capability_detector.detect(effective_prompt)
        previous_backend = self._previous_assistant_backend(
            session=session,
            current_message_id=user_message.message_id,
        )
        if previous_backend:
            request.metadata["previous_backend"] = previous_backend
        tool_result = self.tool_registry.resolve(
            prompt=effective_prompt,
            detection=detection,
            context=runtime_context,
        )
        if tool_result is None:
            detection, tool_result = self._maybe_dispatch_learned_tool(
                prompt=effective_prompt,
                detection=detection,
                context=runtime_context,
                request=request,
            )
        request.metadata.update(self._capability_metadata(detection))
        request.metadata.update(self._phase2_metadata(detection))
        if tool_result is not None and tool_result.success:
            request.metadata.update(self._tool_grounding_metadata(tool_result))
        elif tool_result is not None:
            request.metadata.update(self._tool_pass_through_metadata(tool_result))
        elif self._has_unconfigured_live_capability(detection):
            request.metadata.update(self._pass_through_metadata(detection))

        request = self.compression.compress(request)
        decision = self.route(request, forced_backend=backend)
        request.metadata.update(self._route_metadata(decision))
        if request.metadata.get("private_mode_would_block") and not decision.forced_backend:
            response = SwitchboardResponse(
                request_id=request.request_id,
                backend=decision.backend,
                session_id=session.session_id,
                success=False,
                error_message=(
                    "Local model unavailable; private mode flagged this prompt as "
                    "personal, so Switchboard will not send it to a subscription "
                    "backend. Start Ollama and retry, or redact the prompt."
                ),
                routing_reason=(
                    f"{decision.routing_reason} Private mode blocked subscription fallback."
                ),
                cost_type=BackendCostType.LOCAL,
                estimated_cost_usd=0.0,
            )
            self._finalize_response_metadata(request, response)
            self._record(request, response)
            return response
        adapter = self.registry.get(decision.backend)
        if adapter is None:
            response = SwitchboardResponse(
                request_id=request.request_id,
                backend=decision.backend,
                session_id=session.session_id,
                success=False,
                error_message=(
                    "No configured Switchboard model is available. Install Codex, "
                    "Claude Code, or Ollama and try again."
                ),
                routing_reason=decision.routing_reason,
            )
            if tool_result is not None and not decision.forced_backend:
                self._attach_tool_grounded_answer(
                    request=request,
                    response=response,
                    session_id=session.session_id,
                    prompt=prompt,
                    tool_result=tool_result,
                    base_reason=decision.routing_reason,
                )
            self._finalize_response_metadata(request, response)
            self._record(request, response)
            return response
        if not adapter.is_available():
            response = SwitchboardResponse(
                request_id=request.request_id,
                backend=decision.backend,
                session_id=session.session_id,
                success=False,
                error_message=f"{backend_display_name(decision.backend)} is unavailable.",
                routing_reason=decision.routing_reason,
                cost_type=adapter.cost_type,
                estimated_cost_usd=0.0,
            )
            # This is the path production actually hits when every backend is
            # down (the registry always has adapters, so `adapter is None`
            # above never fires for real). If a deterministic tool already
            # computed the answer, return it instead of discarding it
            # (round-4 dogfood: "what time is it in tokyo" with all backends
            # down returned "Ollama is unavailable." despite the time tool
            # having the answer).
            if (
                tool_result is not None
                and tool_result.success
                and not decision.forced_backend
            ):
                self._attach_tool_grounded_answer(
                    request=request,
                    response=response,
                    session_id=session.session_id,
                    prompt=prompt,
                    tool_result=tool_result,
                    base_reason=decision.routing_reason,
                )
            self._finalize_response_metadata(request, response)
            self._record(request, response)
            return response

        if self._blocked_by_private_mode(request, decision.backend):
            if not decision.forced_backend and not self._is_available("ollama"):
                # The route ended on a subscription backend only because the
                # local model is down; never leak a sensitive prompt because
                # of an outage. Honest refusal instead of telling the user to
                # use a backend that is not running.
                blocked_message = (
                    "Local model unavailable; private mode flagged this prompt as "
                    "personal, so Switchboard will not send it to a subscription "
                    "backend. Start Ollama and retry, or redact the prompt."
                )
            else:
                blocked_message = (
                    f"Backend {decision.backend} is blocked by private mode for sensitive "
                    "content. Use --backend ollama or redact the prompt."
                )
            response = SwitchboardResponse(
                request_id=request.request_id,
                backend=decision.backend,
                session_id=session.session_id,
                success=False,
                error_message=blocked_message,
                routing_reason=(
                    f"{decision.routing_reason} Private mode blocked subscription backend."
                ),
                cost_type=adapter.cost_type,
                estimated_cost_usd=0.0,
            )
            self._finalize_response_metadata(request, response)
            self._record(request, response)
            return response

        try:
            backend_request = self._with_shared_context(
                request=request,
                session=session,
                runtime_context=runtime_context,
                current_message_id=user_message.message_id,
                tool_result=tool_result,
            )
            self._snapshot_context_for_feedback(backend_request)
            request.metadata["model_called"] = True
            response = adapter.ask(backend_request)
        except Exception as exc:
            response = SwitchboardResponse(
                request_id=request.request_id,
                backend=decision.backend,
                session_id=session.session_id,
                success=False,
                error_message=f"{decision.backend} backend failed: {type(exc).__name__}: {exc}",
                routing_reason=decision.routing_reason,
                cost_type=adapter.cost_type,
                estimated_cost_usd=0.0,
            )
        response.session_id = session.session_id
        response.routing_reason = decision.routing_reason
        if response.success:
            response.content = self.response_sanitizer.sanitize(
                response.content,
                user_prompt=prompt,
            )
        if response.success and response.content:
            response = self._maybe_escalate_low_confidence_answer(
                request=request,
                backend_request=backend_request,
                response=response,
                decision=decision,
                local_adapter=adapter,
            )
            response.session_id = session.session_id
        self._finalize_response_metadata(request, response)
        if response.success and response.content:
            assistant_message = self._store_assistant_message(
                session_id=session.session_id,
                response=response,
                display_model=self._response_display_model(response),
                metadata={
                    "request_id": request.request_id,
                    "selected_model": response.selected_model or "",
                },
            )
            response.message_id = assistant_message.message_id
            request.metadata["assistant_message_id"] = assistant_message.message_id
        self._record(request, response)
        return response

    def _maybe_escalate_low_confidence_answer(
        self,
        *,
        request: SwitchboardRequest,
        backend_request: SwitchboardRequest,
        response: SwitchboardResponse,
        decision: BackendRouteDecision,
        local_adapter: AgentAdapter,
    ) -> SwitchboardResponse:
        preferences = self.container.personal_config.preferences
        request.metadata["answer_confidence_escalated"] = False
        if (
            not preferences.escalation_enabled
            or response.backend != "ollama"
            or decision.forced_backend
            or request.metadata.get("sticky_followup")
            or request.metadata.get("answer_confidence_checked")
        ):
            return response
        threshold = preferences.escalation_confidence_threshold
        request.metadata["answer_confidence_checked"] = True
        result = self.answer_confidence.check(
            adapter=local_adapter,
            request=request,
            answer=response.content or "",
            threshold=threshold,
            selected_model=response.selected_model,
        )
        self._attach_confidence_metadata(
            request=request,
            result=result,
            threshold=threshold,
        )
        if result.unavailable or result.passed:
            return response

        sensitive = self._content_is_sensitive(request)
        target = self._confidence_escalation_target(request, decision)
        request.metadata["answer_confidence_escalation_target"] = target
        if sensitive or request.metadata.get("private_mode_would_block"):
            request.metadata["answer_confidence_sensitive_blocked"] = True
            note = (
                "\n\nNote: Switchboard's local confidence check was low, but private "
                "mode keeps this request on the local model instead of escalating it "
                "to a subscription backend."
            )
            original_content = response.content or ""
            original_stdout = response.stdout or original_content
            response.content = f"{original_content}{note}"
            response.stdout = f"{original_stdout}{note}"
            response.routing_reason = (
                f"{response.routing_reason} Local confidence was low; private mode "
                "blocked premium escalation."
            )
            return response
        if target is None or not self._is_available(target):
            request.metadata["answer_confidence_escalation_unavailable"] = True
            return response

        target_adapter = self.registry.get(target)
        if target_adapter is None:
            request.metadata["answer_confidence_escalation_unavailable"] = True
            return response
        escalation_request = backend_request.model_copy(
            update={
                "model": None,
                "metadata": {
                    **backend_request.metadata,
                    "answer_confidence_escalation": True,
                    "escalated_from_backend": response.backend,
                    "escalation_target": target,
                },
            }
        )
        try:
            escalated = target_adapter.ask(escalation_request)
        except Exception as exc:
            request.metadata["answer_confidence_escalation_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return response
        request.metadata["answer_confidence_added_latency_ms"] = (
            result.latency_ms + escalated.latency_ms
        )
        if not escalated.success:
            request.metadata["answer_confidence_escalation_error"] = (
                escalated.error_message or "escalation backend failed"
            )
            return response
        escalated.content = self.response_sanitizer.sanitize(
            escalated.content,
            user_prompt=request.prompt,
        )
        escalated.routing_reason = (
            f"{decision.routing_reason} Local answer confidence score "
            f"{result.score:.2f} was below {threshold:.2f}; escalated to {target}."
        )
        request.metadata.update(
            {
                "answer_confidence_escalated": True,
                "answer_confidence_escalated_from": response.backend,
                "answer_confidence_escalated_to": target,
                "answer_confidence_original_model": response.selected_model or "",
                "answer_confidence_added_latency_ms": result.latency_ms
                + escalated.latency_ms,
            }
        )
        return escalated

    def _attach_confidence_metadata(
        self,
        *,
        request: SwitchboardRequest,
        result: AnswerConfidenceResult,
        threshold: float,
    ) -> None:
        request.metadata.update(
            {
                "answer_confidence_score": result.score,
                "answer_confidence_threshold": threshold,
                "answer_confidence_passed": result.passed,
                "answer_confidence_latency_ms": result.latency_ms,
                "answer_confidence_unavailable": result.unavailable,
            }
        )
        if result.verdict:
            request.metadata["answer_confidence_verdict"] = result.verdict[:32]
        if result.error:
            request.metadata["answer_confidence_error"] = result.error

    def _confidence_escalation_target(
        self,
        request: SwitchboardRequest,
        decision: BackendRouteDecision,
    ) -> str | None:
        detected = set(request.metadata.get("detected_capabilities") or [])
        if decision.route_type == "coding" or Capability.CODING.value in detected:
            return "codex"
        return "claude-code"

    def _resolve_followup_prompt(
        self,
        *,
        prompt: str,
        session: ChatSessionRead,
        current_message_id: str,
    ) -> tuple[str, str | None]:
        if not self._is_retry_prompt(prompt):
            return prompt, None
        recent_messages = self.context_store.get_recent_messages(
            session.session_id,
            limit=self.context_builder.max_recent_messages + 1,
        )
        for message in reversed(recent_messages):
            if message.message_id == current_message_id or message.role != "user":
                continue
            if self._is_retry_prompt(message.content):
                continue
            detection = self.capability_detector.detect(message.content)
            if self._can_reuse_followup_intent(detection):
                return message.content, message.message_id
        return prompt, None

    def _is_retry_prompt(self, prompt: str) -> bool:
        normalized = " ".join(prompt.lower().strip().split())
        return normalized in {
            "again",
            "retry",
            "retry please",
            "try again",
            "try that again",
            "rerun",
            "rerun it",
            "refresh",
            "refresh it",
            "update it",
        }

    def _can_reuse_followup_intent(self, detection: CapabilityDetection) -> bool:
        return any(
            detection.has(capability)
            for capability in (
                Capability.CURRENT_DATE,
                Capability.CURRENT_TIME,
                Capability.LATEST_INFO,
                Capability.STOCK_PRICE,
                Capability.WEATHER,
                Capability.WEB_SEARCH,
            )
        )

    def _capability_metadata(self, detection: CapabilityDetection) -> dict[str, object]:
        return {
            "detected_capabilities": detection.values(),
            "primary_capability": detection.primary.value,
            "tool_used": False,
            "answered_by_tool": False,
            "grounded_by_tool": False,
            "model_called": False,
            "runtime_context_injected": False,
            "tool_available": False,
            "pass_through_to_model": False,
            "web_search_configured": False,
            "web_search_used": False,
        }

    def _tool_grounding_metadata(
        self,
        tool_result: ToolResult,
    ) -> dict[str, object]:
        return {
            "tool_used": True,
            "tool_name": tool_result.tool_name,
            "tool_capability": tool_result.capability.value,
            "answered_by_tool": False,
            "grounded_by_tool": True,
            "tool_available": True,
            "pass_through_to_model": False,
            **tool_result.metadata,
        }

    def _tool_pass_through_metadata(self, tool_result: ToolResult) -> dict[str, object]:
        return {
            "tool_used": False,
            "tool_name": tool_result.tool_name,
            "tool_capability": tool_result.capability.value,
            "answered_by_tool": False,
            "grounded_by_tool": False,
            "tool_available": False,
            "pass_through_to_model": True,
            **tool_result.metadata,
        }

    def _pass_through_metadata(self, detection: CapabilityDetection) -> dict[str, object]:
        return {
            "tool_available": False,
            "pass_through_to_model": True,
            "unconfigured_capabilities": [
                capability.value
                for capability in detection.capabilities
                if capability
                in {
                    Capability.WEATHER,
                    Capability.LATEST_INFO,
                    Capability.STOCK_PRICE,
                    Capability.WEB_SEARCH,
                }
            ],
        }

    def _has_unconfigured_live_capability(self, detection: CapabilityDetection) -> bool:
        return (
            detection.has(Capability.WEATHER)
            or detection.has(Capability.LATEST_INFO)
            or detection.has(Capability.STOCK_PRICE)
            or detection.has(Capability.WEB_SEARCH)
        )

    def _phase2_metadata(self, detection: CapabilityDetection) -> dict[str, object]:
        truth_grounding_needed = self._has_unconfigured_live_capability(detection) or detection.has(
            Capability.CURRENT_TIME
        ) or detection.has(Capability.CURRENT_DATE)
        specialized_tool_available = False
        if detection.has(Capability.CURRENT_TIME) or detection.has(Capability.CURRENT_DATE):
            specialized_tool_available = True
        if detection.has(Capability.STOCK_PRICE):
            specialized_tool_available = (
                self.tool_registry.stock_price_tool.provider.is_configured()
            )
        return {
            "truth_grounding_needed": truth_grounding_needed,
            "specialized_tool_available": specialized_tool_available,
            "web_search_configured": self.tool_registry.web_search_tool.is_configured(),
        }

    # Capabilities the regex detector handles itself; if any is present the
    # learned dispatcher stays out of the way (regex keeps its precision).
    _DISPATCHABLE_CAPABILITIES = frozenset(
        {
            Capability.CURRENT_TIME,
            Capability.CURRENT_DATE,
            Capability.CALCULATION,
            Capability.UNIT_CONVERSION,
            Capability.STOCK_PRICE,
            Capability.LATEST_INFO,
            Capability.WEATHER,
            Capability.WEB_SEARCH,
        }
    )
    # Never dispatch a tool when the prompt is really about code, analysis,
    # or private matters ("write a script that prints the date").
    _DISPATCH_BLOCKERS = frozenset(
        {Capability.CODING, Capability.REASONING, Capability.LOCAL_PRIVATE}
    )

    def _maybe_dispatch_learned_tool(
        self,
        *,
        prompt: str,
        detection: CapabilityDetection,
        context: RuntimeContext,
        request: SwitchboardRequest,
    ) -> tuple[CapabilityDetection, ToolResult | None]:
        """Second-chance tool dispatch when the regexes found nothing.

        The classifier proposes a tool; the tool itself is the judge. For
        deterministic tools the prediction counts only if execution succeeds
        (calculator parses, ticker resolves). Live classes (news, weather)
        adopt the capability so the existing live-data policy handles them
        honestly. Any failure leaves detection exactly as the regexes saw it.
        """
        if self.tool_dispatcher is None:
            return detection, None
        capabilities = set(detection.capabilities)
        if capabilities & (self._DISPATCHABLE_CAPABILITIES | self._DISPATCH_BLOCKERS):
            return detection, None
        result = self.tool_dispatcher.classify(prompt)
        request.metadata["tool_dispatcher_class"] = result.tool_class
        request.metadata["tool_dispatcher_confidence"] = round(result.confidence, 3)
        if not result.success or result.capability is None:
            return detection, None
        candidate = CapabilityDetection(
            capabilities=[
                capability
                for capability in detection.capabilities
                if capability != Capability.UNKNOWN
            ]
            + [result.capability],
            primary=(
                result.capability
                if detection.primary == Capability.UNKNOWN
                else detection.primary
            ),
        )
        candidate_result = self.tool_registry.resolve(
            prompt=prompt,
            detection=candidate,
            context=context,
        )
        if result.tool_class in VERIFIED_TOOL_CLASSES and (
            candidate_result is None or not candidate_result.success
        ):
            # Verification failed: behave as if the dispatcher never fired.
            return detection, None
        request.metadata["tool_dispatcher_used"] = True
        return candidate, candidate_result

    def _attach_tool_grounded_answer(
        self,
        *,
        request: SwitchboardRequest,
        response: SwitchboardResponse,
        session_id: str,
        prompt: str,
        tool_result: ToolResult,
        base_reason: str,
    ) -> None:
        """Return the tool-computed answer directly when no model backend can
        format it: sanitized, free (local cost), honest routing reason, and
        stored in the session like any assistant turn."""
        response.content = self.response_sanitizer.sanitize(
            tool_result.answer,
            user_prompt=prompt,
        )
        response.success = tool_result.success
        response.error_message = tool_result.error
        response.cost_type = BackendCostType.LOCAL
        response.estimated_cost_usd = 0.0
        response.routing_reason = (
            f"{base_reason} No model backend was available, "
            "so Switchboard returned trusted grounding directly."
        )
        assistant_message = self._store_assistant_message(
            session_id=session_id,
            response=response,
            display_model="Switchboard",
            tool_name=tool_result.tool_name,
            metadata={"capability": tool_result.capability.value},
        )
        response.message_id = assistant_message.message_id
        request.metadata["assistant_message_id"] = assistant_message.message_id

    def _finalize_response_metadata(
        self,
        request: SwitchboardRequest,
        response: SwitchboardResponse,
    ) -> None:
        request.metadata.update(
            {
                "success": response.success,
                "failure": not response.success,
                "latency_ms": response.latency_ms,
            }
        )

    def _store_assistant_message(
        self,
        *,
        session_id: str,
        response: SwitchboardResponse,
        display_model: str,
        tool_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ChatMessageRead:
        return self.context_store.append_message(
            session_id=session_id,
            role="assistant",
            content=response.content or response.error_message or "",
            display_model=display_model,
            backend=response.backend,
            tool_name=tool_name,
            metadata=metadata,
        )

    def _response_display_model(self, response: SwitchboardResponse) -> str:
        if response.backend in {"switchboard", "time"} and response.selected_model:
            return response.selected_model
        return backend_display_name(response.backend)

    def _with_shared_context(
        self,
        *,
        request: SwitchboardRequest,
        session: ChatSessionRead,
        runtime_context: RuntimeContext,
        current_message_id: str,
        tool_result: ToolResult | None = None,
    ) -> SwitchboardRequest:
        recent_messages = self.context_store.get_recent_messages(
            session.session_id,
            limit=self.context_builder.max_recent_messages + 1,
        )
        # Elliptical follow-ups to a live-data question ("and microsoft?"
        # after "what is tesla trading at") stick with the previous model but
        # carry no live capability of their own, so the honesty fact would be
        # dropped and the local model would be free to invent a price. Carry
        # it forward deterministically (dogfood regression 2026-06-12).
        if (
            request.metadata.get("sticky_followup")
            and not request.metadata.get("pass_through_to_model")
            and not request.metadata.get("grounded_by_tool")
            and self._previous_turn_had_live_capability(
                recent_messages,
                current_message_id=current_message_id,
            )
        ):
            request.metadata["followup_live_data_honesty"] = True
        memory_facts = self._memory_facts(request)
        context_result = self.context_builder.build(
            session=session,
            recent_messages=recent_messages,
            runtime_context=runtime_context,
            current_request=request.prompt,
            current_message_id=current_message_id,
            trusted_facts=self._trusted_facts(tool_result, request),
            memory_facts=memory_facts,
        )
        request.metadata.update(
            {
                "runtime_context_injected": False,
                "context_injected": True,
                "context_message_count": context_result.message_count,
                "context_summary_used": context_result.summary_used,
                "context_recent_message_count": context_result.recent_message_count,
                "memory_retrieved_count": len(memory_facts),
                "semantic_memory_used": bool(memory_facts),
            }
        )
        # Headroom compression at the context boundary: compress the entire
        # assembled context (history + memory + trusted facts + request),
        # which is what actually reaches the model.
        final_prompt, context_compression_stats = self.compression.compress_context(
            context_result.prompt
        )
        request.metadata.update(context_compression_stats)
        metadata = dict(request.metadata)
        return request.model_copy(update={"prompt": final_prompt, "metadata": metadata})

    def _snapshot_context_for_feedback(self, backend_request: SwitchboardRequest) -> None:
        """Keep the assembled context briefly so a later thumbs-down can store
        exactly what the model saw. Opt-in and capped (see feedback_loop).

        Sensitivity-flagged requests (private-mode reroute or learned
        escalation) are never snapshotted: their assembled context — SSNs,
        health disclosures, and whatever history surrounds them — must not
        persist on disk just because feedback storage is enabled. A later
        thumbs-down on such a request stores an example with empty context;
        only an explicit "wrong model" correction contributes its prompt to
        router training, as the user's deliberate choice (see
        app/api/ui.py:_store_feedback_example).
        """
        if not self.container.personal_config.preferences.store_feedback_examples:
            return
        if backend_request.metadata.get("private_mode_rerouted") or backend_request.metadata.get(
            "sensitivity_escalated"
        ):
            return
        try:
            from switchboard.training.feedback_loop import FeedbackExampleStore

            FeedbackExampleStore(self.container.memory_repository.engine).save_recent_context(
                backend_request.request_id, backend_request.prompt
            )
        except Exception:  # snapshotting must never break the request path
            pass

    def _memory_facts(self, request: SwitchboardRequest) -> list[str]:
        if self.semantic_memory is None:
            return []
        try:
            return self.semantic_memory.retrieve_facts(
                project=request.project,
                prompt=request.prompt,
            )
        except Exception:
            return []

    LIVE_DATA_HONESTY_FACT = (
        "Switchboard has no live-data provider configured for this request "
        "(weather, news, stock prices, or web search). Do not invent specific "
        "facts, figures, headlines, or prices. If you have a real search or "
        "browsing tool available right now, use it; otherwise state plainly that "
        "you cannot access live data and suggest reliable sources. Never ask the "
        "user to grant tool permissions."
    )

    # Injected instead of the honesty fact when the request was routed to
    # Claude Code specifically because its WebSearch tool is enabled. Without
    # this directive Claude tends to answer from memory with a "no live
    # access" disclaimer — spending premium quota on a non-answer.
    CLAUDE_WEB_SEARCH_FACT = (
        "This is a live-data request. Your WebSearch tool is available and "
        "pre-approved for this session: use WebSearch now to look up the "
        "current information before answering. Do not claim you cannot access "
        "live data, and do not answer with generic source suggestions instead "
        "of searching."
    )

    _LIVE_CAPABILITIES = frozenset({"weather", "latest_info", "stock_price", "web_search"})

    def _trusted_facts(
        self,
        tool_result: ToolResult | None,
        request: SwitchboardRequest | None = None,
    ) -> list[str]:
        facts: list[str] = []
        if tool_result is not None and tool_result.success:
            facts.append(tool_result.answer)
            if tool_result.tool_name == "stock_price":
                facts.append(
                    "When answering this stock quote, include the finance source "
                    "and whether the quote may be delayed."
                )
        needs_live_honesty = False
        if request is not None:
            if request.metadata.get("pass_through_to_model"):
                detected = set(request.metadata.get("detected_capabilities") or [])
                # The live-data honesty instruction only applies to live-data
                # questions, not to e.g. failed calculator parses.
                needs_live_honesty = bool(detected & self._LIVE_CAPABILITIES)
            # Sticky follow-ups to a live-data turn inherit the instruction.
            needs_live_honesty = needs_live_honesty or bool(
                request.metadata.get("followup_live_data_honesty")
            )
        if request is not None and needs_live_honesty:
            routed_for_web_search = (
                request.metadata.get("selected_backend") == "claude-code"
                and self.container.personal_config.preferences.claude_code_web_search
            )
            facts.append(
                self.CLAUDE_WEB_SEARCH_FACT
                if routed_for_web_search
                else self.LIVE_DATA_HONESTY_FACT
            )
        return facts

    def _previous_turn_had_live_capability(
        self,
        messages: list[ChatMessageRead],
        *,
        current_message_id: str,
    ) -> bool:
        """True when the nearest substantive earlier user turn asked a
        live-data question. Earlier short follow-ups ("and microsoft?",
        "and apple?") are skipped so chained follow-ups stay covered."""
        for message in reversed(messages):
            if message.message_id == current_message_id or message.role != "user":
                continue
            detection = self.capability_detector.detect(message.content)
            if set(detection.values()) & self._LIVE_CAPABILITIES:
                return True
            if not self._is_short_followup(message.content):
                return False
        return False

    def _route_metadata(self, decision: BackendRouteDecision) -> dict[str, object]:
        return {
            "selected_backend": decision.selected_backend or decision.backend,
            "display_model": decision.display_model,
            "route_type": decision.route_type,
            "routing_reason": decision.routing_reason,
            "fallback_used": decision.fallback_used,
            "fallback_from": decision.fallback_from,
            "forced_backend": decision.forced_backend,
        }

    def route(
        self,
        request: SwitchboardRequest,
        *,
        forced_backend: str | None = None,
    ) -> BackendRouteDecision:
        if forced_backend and forced_backend != "auto":
            return self._decision(
                backend=forced_backend,
                route_type="forced",
                routing_reason=f"User selected backend {forced_backend}.",
                forced_backend=True,
            )

        route_type, preferred, reason = self._classified_route(request)
        if request.metadata.get("private_mode_would_block"):
            return self._decision(
                backend=preferred,
                route_type=route_type,
                routing_reason=reason,
            )
        route_type, preferred, reason = self._quota_adjusted_route(
            request,
            route_type=route_type,
            preferred=preferred,
            reason=reason,
        )
        if request.metadata.get("quota_force_local"):
            return self._decision(
                backend=preferred,
                route_type=route_type,
                routing_reason=reason,
            )
        if self._is_available(preferred):
            return self._decision(
                backend=preferred,
                route_type=route_type,
                routing_reason=reason,
            )

        for candidate in self._fallback_order(route_type):
            if self._is_available(candidate):
                return self._decision(
                    backend=candidate,
                    route_type=route_type,
                    routing_reason=self._fallback_reason(
                        reason,
                        preferred=preferred,
                        candidate=candidate,
                        grounded_by_tool=bool(request.metadata.get("grounded_by_tool")),
                    ),
                    fallback_used=True,
                    fallback_from=preferred,
                )

        return self._decision(
            backend=preferred,
            route_type=route_type,
            routing_reason=f"{reason} No configured backend is currently available.",
            fallback_used=True,
            fallback_from=preferred,
        )

    def _quota_adjusted_route(
        self,
        request: SwitchboardRequest,
        *,
        route_type: str,
        preferred: str,
        reason: str,
    ) -> tuple[str, str, str]:
        if preferred not in PREMIUM_BACKENDS:
            return route_type, preferred, reason
        preferred_status = self.quota_ledger.status_for_backend(preferred)
        if (
            preferred_status is None
            or preferred_status.budget is None
            or not preferred_status.constrained
        ):
            return route_type, preferred, reason

        other = "claude-code" if preferred == "codex" else "codex"
        other_status = self.quota_ledger.status_for_backend(other)
        self._attach_quota_metadata(
            request,
            original_backend=preferred,
            preferred_status=preferred_status,
            alternate_status=other_status,
        )
        other_plausible = other in self._fallback_order(route_type)
        if (
            other_status is not None
            and not other_status.constrained
            and other_plausible
            and self._is_available(other)
        ):
            code = "QUOTA_ALTERNATE_PREMIUM_SELECTED"
            self._mark_quota_decision(request, selected_backend=other, reason_code=code)
            return (
                route_type,
                other,
                (
                    f"{reason} {backend_display_name(preferred)} is at/over the "
                    "user-declared soft quota "
                    f"({preferred_status.used}/{preferred_status.budget} calls in the "
                    f"trailing {preferred_status.window}); using "
                    f"{backend_display_name(other)} instead."
                ),
            )

        if other_status is not None and other_status.constrained:
            code = "QUOTA_BOTH_PREMIUM_CONSTRAINED_LOCAL"
        else:
            code = "QUOTA_PREMIUM_CONSTRAINED_LOCAL"
        request.metadata["quota_force_local"] = True
        self._mark_quota_decision(request, selected_backend="ollama", reason_code=code)
        return (
            route_type,
            "ollama",
            (
                f"{reason} {backend_display_name(preferred)} is at/over the "
                "user-declared soft quota "
                f"({preferred_status.used}/{preferred_status.budget} calls in the "
                f"trailing {preferred_status.window}); using the local model instead "
                "of spending more premium quota."
            ),
        )

    def _attach_quota_metadata(
        self,
        request: SwitchboardRequest,
        *,
        original_backend: str,
        preferred_status: QuotaWindowStatus,
        alternate_status: QuotaWindowStatus | None,
    ) -> None:
        preferred_payload = preferred_status.to_dict()
        alternate_payload = (
            alternate_status.to_dict() if alternate_status is not None else None
        )
        request.metadata.update(
            {
                "quota_routing_influenced": True,
                "quota_original_backend": original_backend,
                "quota_preferred_window": preferred_payload,
                "quota_alternate_window": alternate_payload,
            }
        )

    def _mark_quota_decision(
        self,
        request: SwitchboardRequest,
        *,
        selected_backend: str,
        reason_code: str,
    ) -> None:
        request.metadata.update(
            {
                "quota_selected_backend": selected_backend,
                "quota_reason_code": reason_code,
                "quota_reason_codes": [reason_code],
            }
        )

    def _decision(
        self,
        *,
        backend: str,
        route_type: str,
        routing_reason: str,
        fallback_used: bool = False,
        fallback_from: str | None = None,
        forced_backend: bool = False,
    ) -> BackendRouteDecision:
        return BackendRouteDecision(
            backend=backend,
            selected_backend=backend,
            display_model=backend_display_name(backend),
            routing_reason=routing_reason,
            route_type=route_type,
            fallback_used=fallback_used,
            fallback_from=fallback_from,
            forced_backend=forced_backend,
        )

    def _classified_route(self, request: SwitchboardRequest) -> tuple[str, str, str]:
        """Classify a request using the configured router mode.

        Returns (route_type, preferred_backend, routing_reason) and records
        router telemetry in request.metadata. Deterministic rules are always
        the fallback when the LLM router is unavailable or unparseable.
        """
        # Privacy first: sensitive content stays on the local model instead of
        # falling through to subscription fallback when the local model is down.
        if self._content_is_sensitive(request):
            request.metadata["private_mode_rerouted"] = True
            if not self._is_available("ollama"):
                request.metadata["private_mode_would_block"] = True
                return (
                    "local",
                    "ollama",
                    "Private mode flagged this prompt as sensitive; the local model "
                    "is unavailable, so Switchboard would refuse it rather than send "
                    "it to Claude or Codex.",
                )
            return (
                "local",
                "ollama",
                "Private mode detected sensitive content; keeping this request on the "
                "local model.",
            )

        # Tool-grounded answers only need formatting; never spend premium
        # quota on formatting a trusted fact (unless the prompt also needs
        # coding or reasoning work).
        detected = set(request.metadata.get("detected_capabilities") or [])
        if (
            request.metadata.get("grounded_by_tool")
            and not ({"coding", "reasoning"} & detected)
            and self._is_available("ollama")
        ):
            return (
                "local",
                "ollama",
                "A deterministic tool grounded the answer; the free local model "
                "formats it.",
            )

        # Live-data questions without a configured provider get the same
        # disclaimer from every model, so never spend premium quota on them.
        if request.metadata.get("pass_through_to_model") and (
            detected & self._LIVE_CAPABILITIES
        ):
            preferences = self.container.personal_config.preferences
            if preferences.claude_code_web_search and self._is_available("claude-code"):
                return (
                    "reasoning",
                    "claude-code",
                    "Live-data request; Claude Code web search is enabled, so Claude "
                    "can look this up.",
                )
            if self._is_available("ollama"):
                request.metadata["live_data_rerouted_local"] = True
                return (
                    "local",
                    "ollama",
                    "Live-data request without a configured provider; every model "
                    "would give the same disclaimer, so Switchboard uses the free "
                    "local model.",
                )

        rules_route_type, rules_backend, rules_reason = self._preferred_backend(request.prompt)
        request.metadata.setdefault("router_mode", self.router_mode)
        request.metadata.setdefault("llm_router_used", False)

        # Short follow-ups ("can you do it yourself", "why?", "continue")
        # carry no routable signal; continuity beats reclassification, so they
        # stick with the model that answered the previous turn.
        previous_backend = str(request.metadata.get("previous_backend") or "")
        if (
            rules_route_type == "unknown"
            and previous_backend
            and self._is_short_followup(request.prompt)
            and self._is_available(previous_backend)
        ):
            request.metadata["sticky_followup"] = True
            return (
                "followup",
                previous_backend,
                "Short follow-up; continuing with the same model as the previous turn.",
            )
        # Learned mode: a tiny embedding classifier replaces only the
        # local/coding/reasoning choice. Policy above (privacy, grounding,
        # live-data, forced, stickiness) already ran deterministically.
        if self.router_mode == "learned" and self.learned_router is not None:
            learned = self.learned_router.classify(request.prompt)
            request.metadata.update(
                {
                    "learned_router_used": learned.success,
                    "learned_router_model": learned.model,
                    "learned_router_confidence": learned.confidence,
                    "learned_router_route_type": learned.route_type,
                    "learned_router_latency_ms": learned.latency_ms,
                }
            )
            if learned.success:
                # "tool" with no tool grounded upstream is a cheap local task.
                route_type = "local" if learned.route_type == "tool" else learned.route_type
                backend = "ollama" if learned.route_type == "tool" else learned.backend
                reason = (
                    f"Learned router classified this as {learned.route_type} "
                    f"(confidence {learned.confidence:.2f})."
                )
                return route_type, backend, reason
            request.metadata["learned_router_error"] = learned.error
            return (
                rules_route_type,
                rules_backend,
                f"{rules_reason} Learned router unavailable/low-confidence; used rules.",
            )

        if self.router_mode in {"rules", "learned"} or self.llm_router is None:
            return rules_route_type, rules_backend, rules_reason
        if self.router_mode == "hybrid" and rules_route_type != "unknown":
            request.metadata["llm_router_skipped_reason"] = "rules_confident"
            return rules_route_type, rules_backend, rules_reason
        result = self.llm_router.classify(request.prompt)
        request.metadata.update(
            {
                "llm_router_used": result.success,
                "llm_router_model": result.model,
                "llm_router_latency_ms": result.latency_ms,
                "llm_router_confidence": result.confidence,
            }
        )
        if not result.success:
            request.metadata["llm_router_error"] = result.error
            return (
                rules_route_type,
                rules_backend,
                f"{rules_reason} LLM router unavailable; used deterministic rules.",
            )
        reason = (
            f"LLM router ({result.model}) classified this as a {result.route_type} task "
            f"(confidence {result.confidence:.2f})."
        )
        return result.route_type, result.backend, reason

    def _is_short_followup(self, prompt: str) -> bool:
        return len(prompt.split()) <= 6

    def _previous_assistant_backend(
        self,
        *,
        session: ChatSessionRead,
        current_message_id: str,
    ) -> str | None:
        messages = self.context_store.get_recent_messages(session.session_id, limit=6)
        for message in reversed(messages):
            if message.message_id == current_message_id:
                continue
            if message.role == "assistant" and message.backend in {
                "ollama",
                "codex",
                "claude-code",
            }:
                return message.backend
        return None

    def _preferred_backend(self, prompt: str) -> tuple[str, str, str]:
        text = prompt.lower()
        # Substantive local signal (privacy, summarize/rewrite, explicit
        # "local") outranks everything: cheap-and-private is the product
        # thesis. Bare fillers ("hey", "ok", "quick") are checked LAST so a
        # voice-style "hey, debug this python traceback" still reaches the
        # coding model (dogfood regression 2026-06-12).
        if self._matches_any(text, self._local_keywords()):
            return "local", "ollama", "Detected local/private/simple task; prefers Ollama."
        if self._matches_any(text, self._coding_keywords()):
            return "coding", "codex", "Detected coding/debugging task; prefers Codex."
        if self._matches_any(text, self._reasoning_keywords()):
            return (
                "reasoning",
                "claude-code",
                "Detected architecture/design/reasoning task; prefers Claude Code.",
            )
        if self._matches_any(text, self._filler_keywords()):
            return "local", "ollama", "Detected local/private/simple task; prefers Ollama."
        # Local-first by design: premium backends are a deliberate exception
        # (coding/reasoning signal, explicit user choice, web-search reroute),
        # never the default. Failing open to a subscription model would also
        # leak keyword-free sensitive prompts the learned escalator misses.
        return (
            "unknown",
            "ollama",
            "Unknown task; local-first default keeps it on the free local model.",
        )

    def _fallback_order(self, route_type: str) -> list[str]:
        fallback_by_route_type = {
            "coding": ["codex", "claude-code", "ollama"],
            "reasoning": ["claude-code", "codex", "ollama"],
            "local": ["ollama", "claude-code", "codex"],
            "unknown": ["ollama", "claude-code", "codex"],
        }
        return fallback_by_route_type.get(route_type, fallback_by_route_type["unknown"])

    def _fallback_reason(
        self,
        reason: str,
        *,
        preferred: str,
        candidate: str,
        grounded_by_tool: bool = False,
    ) -> str:
        if grounded_by_tool and preferred == "ollama" and candidate != "ollama":
            # "local-first default keeps it on the free local model" would be
            # misleading here: a tool already grounded the answer and the
            # local model is down, so a premium model only formats facts.
            return (
                "Tool-grounded request; the local model is unavailable, so "
                f"{backend_display_name(candidate)} formats the trusted facts. "
                f"({backend_display_name(preferred)} was unavailable; "
                f"fell back to {backend_display_name(candidate)}.)"
            )
        return (
            f"{reason} {backend_display_name(preferred)} was unavailable; "
            f"fell back to {backend_display_name(candidate)}."
        )

    def _matches_any(self, text: str, patterns: tuple[str, ...]) -> bool:
        for pattern in patterns:
            if " " in pattern:
                if pattern in text:
                    return True
                continue
            if re.search(rf"\b{re.escape(pattern)}\b", text):
                return True
        return False

    def _coding_keywords(self) -> tuple[str, ...]:
        return (
            "code",
            "coding",
            "bug",
            "debug",
            "test",
            "tests",
            "failing",
            "failure",
            "implement",
            "refactor",
            "repo",
            "repository",
            "file",
            "function",
            "class",
            "pr",
            "pull request",
            "diff",
            "git",
            "error",
            "traceback",
            "stack trace",
            "compile",
            "build",
            "package",
            "dependency",
            "pytest",
            "unittest",
            "prompt for codex",
            "ui code",
            "run code",
            "update code",
            # Programming languages and CS staples routed to the coding agent.
            "java",
            "python",
            "javascript",
            "typescript",
            "c++",
            "golang",
            "rust",
            "kotlin",
            "swift",
            "sql",
            "bash",
            "regex",
            "algorithm",
            "data structure",
            "linked list",
            "binary tree",
            "hash map",
            "leetcode",
            "script",
            # Web/app development vocabulary (dogfood: "create me a project
            # that has a login page" was mis-routed to a small local model,
            # which produced insecure code).
            "login page",
            "signup page",
            "web app",
            "webapp",
            "website",
            "web page",
            "webpage",
            "frontend",
            "backend",
            "html",
            "css",
            "database",
            "rest api",
            "create a project",
            "create me a project",
            "build a project",
            "build me a project",
            "build an app",
            "build me an app",
        )

    def _reasoning_keywords(self) -> tuple[str, ...]:
        return (
            "architecture",
            "design",
            "system design",
            "tradeoff",
            "tradeoffs",
            "compare",
            "explain",
            "strategy",
            "plan",
            "research",
            "paper",
            "patent",
            "principal engineer",
            "product manager",
            "review",
            "evaluate",
            "reasoning",
            "distributed system",
            "distributed systems",
            "scalability",
            "reliability",
        )

    def _local_keywords(self) -> tuple[str, ...]:
        # Substantive local/private/simple-task signal. Checked FIRST in
        # _preferred_backend, so keep pure conversational fillers out of this
        # list (they live in _filler_keywords and only fire when no coding or
        # reasoning signal exists).
        return (
            "local",
            "locally",
            "private",
            "privacy",
            "cheap",
            "simple",
            "summarize",
            "summarise",
            "summary",
            "rewrite",
            "offline",
            "who are you",
            "what can you do",
        )

    def _filler_keywords(self) -> tuple[str, ...]:
        # Greetings, acknowledgements, and brevity fillers. These mark small
        # talk only when nothing else matched: "hey can you debug this python
        # traceback" must reach the coding model, while a bare "hey" or
        # "thanks" stays free and local.
        return (
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "quick",
            "short",
            "how are you",
            "good morning",
            "good evening",
            "good night",
            "what's up",
            "whats up",
        )

    def _is_available(self, backend: str) -> bool:
        adapter = self.registry.get(backend)
        return bool(adapter and adapter.is_available())

    def _blocked_by_private_mode(self, request: SwitchboardRequest, backend: str) -> bool:
        if backend == "ollama" or not self.container.personal_config.preferences.private_mode:
            return False
        return self._content_is_sensitive(request)

    def _content_is_sensitive(self, request: SwitchboardRequest) -> bool:
        """Keyword sensitivity first (the floor, final when positive); the
        learned escalator may only ADD protection when keywords found
        nothing. Escalator failures of any kind leave the keyword verdict."""
        if not self.container.personal_config.preferences.private_mode:
            return False
        if self._keyword_sensitive(request):
            return True
        if self.sensitivity_escalator is None:
            return False
        escalation = self.sensitivity_escalator.classify(request.prompt)
        if escalation.success and escalation.escalate:
            request.metadata["sensitivity_escalated"] = True
            request.metadata["sensitivity_escalation_confidence"] = round(
                escalation.confidence, 3
            )
            return True
        return False

    def _keyword_sensitive(self, request: SwitchboardRequest) -> bool:
        classification = self.container.classifier.classify(
            NormalizedRequest(
                request_id=request.request_id,
                tenant_id="local-user",
                application_id="switchboard-core",
                workflow_id=request.project,
                environment="local",
                messages=[ChatMessage(role="user", content=request.prompt)],
                input_token_estimate=self.container.cost_estimator.estimate_text_tokens(
                    request.prompt
                ),
                requested_model="auto",
                metadata=request.metadata,
                routing_mode=RoutingMode.ACTIVE,
                created_at=request.created_at,
            )
        )
        return classification.sensitivity in {
            Sensitivity.CONFIDENTIAL,
            Sensitivity.REGULATED,
            Sensitivity.PRIVATE_PERSONAL,
        }

    def _record(self, request: SwitchboardRequest, response: SwitchboardResponse) -> None:
        self.metrics.add(
            BackendMetricRecord(
                request_id=response.request_id,
                backend=response.backend,
                selected_model=response.selected_model,
                project=request.project,
                prompt_char_count=request.prompt_char_count,
                latency_ms=response.latency_ms,
                success=response.success,
                error_message=sanitize_provider_error(
                    response.error_message,
                    prompt=request.prompt,
                    backend=response.backend,
                ),
                exit_code=response.exit_code,
                routing_reason=response.routing_reason,
                cost_type=response.cost_type.value,
                estimated_cost_usd=response.estimated_cost_usd,
                private_mode=request.private_mode,
                metadata_json=json.dumps(request.metadata),
            )
        )
