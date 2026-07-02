from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from switchboard import __version__
from switchboard.app.models.backends import SwitchboardResponse, backend_display_name
from switchboard.app.models.personal import FeedbackCreate, FeedbackRead
from switchboard.app.models.sessions import ChatSessionRead
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.core_factory import build_configured_core_service
from switchboard.app.services.local_runtime import OllamaRuntimeService
from switchboard.app.services.personal_switchboard import PersonalSwitchboardService
from switchboard.app.services.quota import PREMIUM_BACKENDS, QuotaLedgerService
from switchboard.app.services.switchboard_core import SwitchboardCoreService
from switchboard.app.services.update_check import cached_version_status
from switchboard.app.utils.remote import (
    REMOTE_MUTATION_ENV,
    host_is_loopback,
    remote_mutations_allowed,
)

router = APIRouter(tags=["ui"])

BACKEND_BY_UI_VALUE: dict[str, str | None] = {
    "auto": None,
    "codex": "codex",
    "claude": "claude-code",
    "ollama": "ollama",
}

UI_VALUE_BY_BACKEND = {
    "codex": "codex",
    "claude-code": "claude",
    "ollama": "ollama",
}

HTTP_DISABLED_CLI_BACKENDS = {"codex", "claude-code"}
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
REMOTE_MUTATION_DETAIL = (
    "Remote UI mutations are disabled unless "
    f"{REMOTE_MUTATION_ENV}=1 is set for this server."
)


class UiChatRequest(BaseModel):
    message: str = Field(min_length=1)
    backend: str = "auto"
    session_id: str | None = None
    private: bool = False


class UiChatResponse(BaseModel):
    session_id: str
    answer: str
    backend: str
    display_model: str


class UiHistoryMessage(BaseModel):
    message_id: str
    role: str
    content: str
    display_model: str | None = None
    backend: str | None = None
    request_id: str | None = None
    routing: dict[str, object] | None = None
    feedback_rating: str | None = None
    corrected_backend: str | None = None
    created_at: str


class UiHistoryResponse(BaseModel):
    session_id: str
    private: bool = False
    messages: list[UiHistoryMessage]


class UiFeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    rating: str = Field(min_length=1)
    note: str | None = None
    # Thumbs-down disambiguation: "bad_answer" or "wrong_model".
    detail: str | None = None
    corrected_backend: str | None = None  # ollama | codex | claude-code


class UiFeedbackPendingResponse(BaseModel):
    pending: int


class UiSessionPatchRequest(BaseModel):
    title: str | None = None
    private: bool | None = None


class FeedbackAckPayload(TypedDict):
    pending_corrections: int
    ack_message: str
    copy_command: str | None
    nudge_enable_examples: bool


def request_host(request: Request) -> str | None:
    return request.client.host if request.client is not None else None


def require_local_mutation(request: Request) -> None:
    if host_is_loopback(request_host(request)) or remote_mutations_allowed():
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={"message": REMOTE_MUTATION_DETAIL},
    )


def core_service(request: Request) -> SwitchboardCoreService:
    container: ServiceContainer = request.app.state.container
    service = build_configured_core_service(container, cwd=Path.cwd())
    if not http_cli_backends_enabled():
        for backend in HTTP_DISABLED_CLI_BACKENDS:
            service.registry.adapters.pop(backend, None)
    return service


def http_cli_backends_enabled() -> bool:
    return (
        os.getenv("SWITCHBOARD_HTTP_ENABLE_CLI_BACKENDS", "").strip().lower() in TRUTHY_ENV_VALUES
    )


def ui_backend_name(backend: str) -> str:
    return UI_VALUE_BY_BACKEND.get(backend, backend)


def display_model_name(backend: str) -> str:
    return backend_display_name(backend)


def response_display_model_name(response: SwitchboardResponse) -> str:
    if response.backend in {"switchboard", "time"} and response.selected_model:
        return response.selected_model
    return display_model_name(response.backend)


def metric_metadata(request: Request, request_id: str | None) -> dict[str, object]:
    if not request_id:
        return {}
    container: ServiceContainer = request.app.state.container
    metric = container.backend_metrics_repository.get(request_id)
    return dict(metric.metadata) if metric is not None else {}


def int_metadata(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return max(0, int(value))
    return 0


def compression_percent(metadata: dict[str, object]) -> int | None:
    ratio = metadata.get("context_compression_ratio", metadata.get("compression_ratio"))
    if not isinstance(ratio, int | float) or ratio >= 1:
        return None
    return max(0, min(99, round((1 - float(ratio)) * 100)))


def routing_chip_metadata(metadata: dict[str, object]) -> dict[str, object]:
    percent = compression_percent(metadata)
    payload: dict[str, object] = {
        "route_type": str(metadata.get("route_type") or ""),
        "private_chat": bool(metadata.get("private_chat")),
        "privacy_floor": bool(
            metadata.get("private_mode_rerouted")
            or metadata.get("private_mode_would_block")
            or metadata.get("sensitivity_escalated")
        ),
        "tool_grounded": bool(metadata.get("grounded_by_tool")),
        "compressed": bool(
            metadata.get("context_compression_used") or metadata.get("compression_used")
        ),
        "escalated": bool(metadata.get("answer_confidence_escalated")),
        "quota": bool(metadata.get("quota_routing_influenced")),
    }
    if percent is not None:
        payload["compression_percent"] = percent
    if metadata.get("quota_reason_code"):
        payload["quota_reason_code"] = str(metadata["quota_reason_code"])
    return payload


def clean_backend_error(response: SwitchboardResponse) -> str:
    backend = ui_backend_name(response.backend)
    display_name = {
        "codex": "Codex",
        "claude": "Claude",
        "ollama": "Ollama",
    }.get(backend, "The selected backend")
    raw_error = response.error_message or ""
    lower_error = raw_error.lower()

    if "private mode" in lower_error or "sensitive content" in lower_error:
        return (
            "Private mode blocked this request for the selected model. "
            "Choose Ollama or redact sensitive details."
        )
    if "timed out" in lower_error:
        return (
            f"{display_name} timed out. Try a shorter prompt, increase the timeout from "
            "the CLI, or choose another model."
        )
    if "unavailable" in lower_error or "not found" in lower_error:
        if backend == "ollama":
            return "Ollama is not running. Start Ollama or choose another model."
        if backend == "codex":
            return (
                "Codex is not available. Please install and authenticate Codex, "
                "or choose another model."
            )
        if backend == "claude":
            return (
                "Claude is not available. Please install and authenticate Claude Code, "
                "or choose another model."
            )
        return f"{display_name} is not available. Choose another model."
    if "no enabled ollama chat model" in lower_error:
        return "Ollama has no enabled chat model. Install or enable a chat model first."
    if "no configured switchboard model" in lower_error:
        return "No Switchboard model is available. Install Codex, Claude Code, or Ollama."
    return "Something went wrong. Please try again or choose another model."


def validated_message_and_backend(payload: UiChatRequest) -> tuple[str, str]:
    message = payload.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Enter a message before sending."},
        )
    selected_backend = payload.backend.strip().lower()
    if selected_backend not in BACKEND_BY_UI_VALUE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Choose Auto, Codex, Claude, or Ollama."},
        )
    return message, selected_backend


def ask_switchboard(
    payload: UiChatRequest,
    request: Request,
) -> SwitchboardResponse:
    message, selected_backend = validated_message_and_backend(payload)
    container: ServiceContainer = request.app.state.container
    stored_session = (
        container.context_store.get_session(payload.session_id) if payload.session_id else None
    )
    private_chat = payload.private or bool(stored_session and stored_session.private)
    if private_chat:
        selected_backend = "ollama"
    forced_backend = BACKEND_BY_UI_VALUE[selected_backend]
    if forced_backend in HTTP_DISABLED_CLI_BACKENDS and not http_cli_backends_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "message": (
                    "Subscription CLI backends are disabled on the HTTP API by default. "
                    "Use the switchboard CLI locally, choose Ollama, or set "
                    "SWITCHBOARD_HTTP_ENABLE_CLI_BACKENDS=true to opt in."
                )
            },
        )
    response = core_service(request).ask(
        message,
        backend=forced_backend,
        project="ui",
        metadata={
            "surface": "ui",
            "requested_backend": payload.backend.strip().lower(),
            "private_chat": private_chat,
        },
        session_id=payload.session_id,
    )
    if private_chat and response.session_id:
        container.context_store.update_session(response.session_id, private=True)
    return response


def response_payload(response: SwitchboardResponse) -> UiChatResponse:
    return UiChatResponse(
        session_id=response.session_id or "",
        answer=(response.content or "").strip(),
        backend=response.backend,
        display_model=response_display_model_name(response),
    )


def stream_event(event_type: str, **payload: object) -> str:
    return json.dumps({"type": event_type, **payload}) + "\n"


def answer_chunks(answer: str, chunk_size: int = 24) -> Iterator[str]:
    for start in range(0, len(answer), chunk_size):
        yield answer[start : start + chunk_size]


def stream_chat_response(
    response: SwitchboardResponse,
    metadata: dict[str, object] | None = None,
) -> Iterator[str]:
    yield stream_event("start", session_id=response.session_id)
    if not response.success:
        yield stream_event(
            "error",
            message=clean_backend_error(response),
            backend=response.backend,
            display_model=response_display_model_name(response),
            session_id=response.session_id,
        )
        return

    payload = response_payload(response)
    metadata = metadata or {}
    routing_info = {
        "request_id": response.request_id,
        "routing_reason": response.routing_reason,
        "latency_ms": response.latency_ms,
        "cost_type": response.cost_type.value,
        "selected_model": response.selected_model,
        **routing_chip_metadata(metadata),
    }
    yield stream_event(
        "metadata",
        session_id=payload.session_id,
        backend=payload.backend,
        display_model=payload.display_model,
        **routing_info,
    )
    answer = payload.answer or "No answer returned."
    for chunk in answer_chunks(answer):
        yield stream_event("chunk", text=chunk)
    yield stream_event(
        "done",
        session_id=payload.session_id,
        backend=payload.backend,
        display_model=payload.display_model,
        **routing_info,
    )


@router.post("/api/chat", response_model=UiChatResponse)
def chat(payload: UiChatRequest, request: Request) -> UiChatResponse:
    response = ask_switchboard(payload, request)
    if not response.success:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": clean_backend_error(response),
                "backend": response.backend,
                "display_model": response_display_model_name(response),
                "session_id": response.session_id,
            },
        )
    return response_payload(response)


@router.post("/api/chat/stream")
def chat_stream(payload: UiChatRequest, request: Request) -> StreamingResponse:
    response = ask_switchboard(payload, request)
    metadata = metric_metadata(request, response.request_id)
    return StreamingResponse(
        stream_chat_response(response, metadata),
        media_type="application/x-ndjson",
    )


@router.get("/api/chat/history", response_model=UiHistoryResponse)
def chat_history(session_id: str, request: Request) -> UiHistoryResponse:
    container: ServiceContainer = request.app.state.container
    session = container.context_store.get_session(session_id)
    if session is None:
        return UiHistoryResponse(session_id=session_id, messages=[])
    records = container.context_store.list_messages(session_id)
    request_ids = [
        str(record.metadata.get("request_id") or "")
        for record in records
        if record.role == "assistant" and record.metadata.get("request_id")
    ]
    feedback_by_request = container.personal_telemetry_repository.feedback_by_request_ids(
        request_ids
    )
    messages: list[UiHistoryMessage] = []
    for record in records:
        if record.role not in {"user", "assistant"}:
            continue
        request_id = str(record.metadata.get("request_id") or "") or None
        feedback = feedback_by_request.get(request_id or "") if record.role == "assistant" else None
        routing = (
            routing_chip_metadata(metric_metadata(request, request_id))
            if record.role == "assistant"
            else None
        )
        messages.append(
            UiHistoryMessage(
                message_id=record.message_id,
                role=record.role,
                content=record.content,
                display_model=record.display_model,
                backend=record.backend,
                request_id=request_id,
                routing=routing,
                feedback_rating=feedback.rating if feedback is not None else None,
                corrected_backend=feedback.preferred_model if feedback is not None else None,
                created_at=record.created_at.isoformat(),
            )
        )
    return UiHistoryResponse(session_id=session_id, private=session.private, messages=messages)


@router.patch(
    "/api/sessions/{session_id}",
    response_model=ChatSessionRead,
    dependencies=[Depends(require_local_mutation)],
)
def update_ui_session(
    session_id: str,
    payload: UiSessionPatchRequest,
    request: Request,
) -> ChatSessionRead:
    container: ServiceContainer = request.app.state.container
    updated = container.context_store.update_session(
        session_id,
        title=payload.title,
        private=payload.private,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Unknown session."},
        )
    return updated


@router.get("/api/backends/status")
def backends_status(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    service = build_configured_core_service(container, cwd=Path.cwd())
    infos = {backend.name: backend for backend in service.backends()}
    http_enabled = http_cli_backends_enabled()
    loaded = sorted(OllamaRuntimeService(container.personal_config).list_loaded_models())
    options: list[dict[str, object]] = [
        {
            "value": "auto",
            "backend": None,
            "label": "Auto",
            "description": "Routes automatically",
            "available": True,
            "hot": False,
        }
    ]
    for value, backend, description in (
        ("codex", "codex", "Best for coding tasks"),
        ("claude", "claude-code", "Good for reasoning and design"),
        ("ollama", "ollama", "Runs locally"),
    ):
        info = infos.get(backend)
        available = bool(info and info.available)
        disabled_reason = None
        if backend in HTTP_DISABLED_CLI_BACKENDS and not http_enabled:
            available = False
            disabled_reason = "HTTP CLI backend opt-in required"
        hot_models = loaded if backend == "ollama" else []
        options.append(
            {
                "value": value,
                "backend": backend,
                "label": backend_display_name(backend),
                "description": description,
                "available": available,
                "hot": bool(hot_models),
                "hot_models": hot_models,
                "details": info.details if info is not None else None,
                "warning": disabled_reason or (info.warning if info is not None else None),
            }
        )
    return {
        "options": options,
        "loaded_local_models": loaded,
        "private_mode": container.personal_config.preferences.private_mode,
    }


@router.get("/api/dashboard")
def dashboard(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    now = datetime.now(UTC)
    since = now - timedelta(days=7)
    records = container.backend_metrics_repository.list_since(since=since, limit=10000)
    usage_by_backend = Counter(record.backend for record in records)
    successful = [record for record in records if record.success]
    premium_calls = sum(1 for record in successful if record.backend in PREMIUM_BACKENDS)
    tool_calls = sum(
        1
        for record in successful
        if record.backend in {"switchboard", "time"} or record.metadata.get("grounded_by_tool")
    )
    local_calls = sum(
        1
        for record in successful
        if record.backend not in PREMIUM_BACKENDS
        and record.backend not in {"switchboard", "time"}
        and not record.metadata.get("grounded_by_tool")
    )
    premium_avoided = local_calls + tool_calls
    compression_saved = 0
    routing_saved = 0
    for record in records:
        compression_saved += int_metadata(record.metadata, "compression_tokens_saved")
        compression_saved += int_metadata(record.metadata, "context_compression_tokens_saved")
        if record.success and record.backend not in PREMIUM_BACKENDS:
            routing_saved += max(0, round(record.prompt_char_count / 4))

    trend = []
    counts_by_day: dict[str, Counter[str]] = {}
    for record in records:
        key = record.created_at.date().isoformat()
        counts_by_day.setdefault(key, Counter())
        counts_by_day[key]["requests"] += 1
        if record.success and record.backend in PREMIUM_BACKENDS:
            counts_by_day[key]["premium_calls"] += 1
        elif record.success and (
            record.backend in {"switchboard", "time"} or record.metadata.get("grounded_by_tool")
        ):
            counts_by_day[key]["tool_calls"] += 1
        elif record.success:
            counts_by_day[key]["local_calls"] += 1
    for days_ago in range(6, -1, -1):
        day = (now - timedelta(days=days_ago)).date().isoformat()
        counts = counts_by_day.get(day, Counter())
        trend.append(
            {
                "date": day,
                "requests": counts.get("requests", 0),
                "local_calls": counts.get("local_calls", 0),
                "tool_calls": counts.get("tool_calls", 0),
                "premium_calls": counts.get("premium_calls", 0),
            }
        )
    feedback_summary = container.personal_telemetry_repository.feedback_summary()

    return {
        "window_days": 7,
        "total_requests": len(records),
        "premium_calls": premium_calls,
        "premium_calls_avoided_vs_always_premium": premium_avoided,
        "handled_requests": {
            "local": local_calls,
            "tools": tool_calls,
            "premium": premium_calls,
        },
        "estimated_tokens_saved": {
            "compression": compression_saved,
            "routing": routing_saved,
            "total": compression_saved + routing_saved,
        },
        "usage_by_backend": dict(usage_by_backend),
        "last_7_days": trend,
        "feedback": {
            "total": feedback_summary["total"],
            "good": feedback_summary["positive"],
            "bad": feedback_summary["bad"],
            "corrected": feedback_summary["wrong_route"],
            "pending_corrections": feedback_pending_count(container),
        },
    }


@router.get("/api/quota")
def quota_status(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    return QuotaLedgerService(
        container.backend_metrics_repository,
        container.personal_config.quota,
    ).snapshot()


@router.get("/api/version")
def version_status() -> dict[str, object]:
    status = cached_version_status(__version__)
    return {
        "installed": status.installed,
        "latest": status.latest,
        "update_available": status.update_available,
    }


VALID_CORRECTED_BACKENDS = ("ollama", "codex", "claude-code")


def feedback_pending_count(container: ServiceContainer) -> int:
    try:
        from switchboard.training.feedback_loop import FeedbackExampleStore

        return FeedbackExampleStore(
            container.memory_repository.engine
        ).unprocessed_wrong_model_count()
    except Exception:
        return 0


def feedback_ack_payload(
    container: ServiceContainer,
    *,
    rating: str,
    pending: int,
) -> FeedbackAckPayload:
    preferences = container.personal_config.preferences
    if rating != "wrong-route":
        return {
            "pending_corrections": pending,
            "ack_message": "Saved.",
            "copy_command": None,
            "nudge_enable_examples": False,
        }
    if not preferences.store_feedback_examples:
        return {
            "pending_corrections": pending,
            "ack_message": (
                "Saved - this correction immediately nudges routing preferences. "
                "To also retrain the classifier from your corrections, enable "
                "store_feedback_examples."
            ),
            "copy_command": None,
            "nudge_enable_examples": True,
        }
    if preferences.feedback_auto_retrain:
        threshold = preferences.feedback_retrain_threshold
        return {
            "pending_corrections": pending,
            "ack_message": (
                f"Saved - {pending} of {threshold} corrections until the router "
                "retrains automatically."
            ),
            "copy_command": None,
            "nudge_enable_examples": False,
        }
    command = "switchboard train-router"
    return {
        "pending_corrections": pending,
        "ack_message": (
            f"Saved - {pending} corrections pending. Run `{command}` to apply."
            if pending >= 3
            else f"Saved - {pending} corrections pending."
        ),
        "copy_command": command if pending >= 3 else None,
        "nudge_enable_examples": False,
    }


@router.post("/api/chat/feedback", response_model=FeedbackRead)
def chat_feedback(payload: UiFeedbackRequest, request: Request) -> FeedbackRead:
    container: ServiceContainer = request.app.state.container
    rating = payload.rating.strip().lower()
    if rating not in {"good", "too-weak", "wrong-route", "bad"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Rating must be good, too-weak, wrong-route, or bad."},
        )
    detail = (payload.detail or "").strip().lower() or None
    corrected = (payload.corrected_backend or "").strip().lower() or None
    if rating == "wrong-route":
        detail = "wrong_model"
    if rating == "bad":
        detail = detail or "bad_answer"
    if detail == "wrong_model" and corrected not in VALID_CORRECTED_BACKENDS:
        # Reject before anything is stored: a wrong-model verdict without a
        # valid correction cannot train the router but would still count
        # toward the retrain threshold.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": (
                    "corrected_backend must be one of: " + ", ".join(VALID_CORRECTED_BACKENDS) + "."
                )
            },
        )
    result = PersonalSwitchboardService(container).add_feedback(
        FeedbackCreate(
            request_id=payload.request_id.strip(),
            rating=rating,
            note=payload.note,
            preferred_model=corrected if detail == "wrong_model" else None,
        )
    )
    _store_feedback_example(
        container,
        request_id=payload.request_id.strip(),
        rating=rating,
        detail=detail,
        corrected_backend=corrected,
    )
    pending = feedback_pending_count(container)
    ack = feedback_ack_payload(container, rating=rating, pending=pending)
    result.pending_corrections = ack["pending_corrections"]
    result.ack_message = ack["ack_message"]
    result.copy_command = ack["copy_command"]
    result.nudge_enable_examples = ack["nudge_enable_examples"]
    return result


@router.delete(
    "/api/chat/feedback/{request_id}",
    dependencies=[Depends(require_local_mutation)],
)
def delete_chat_feedback(request_id: str, request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    deleted = container.personal_telemetry_repository.delete_feedback(request_id.strip())
    return {
        "request_id": request_id,
        "deleted": deleted,
        "pending": feedback_pending_count(container),
    }


@router.get("/api/feedback/pending", response_model=UiFeedbackPendingResponse)
def feedback_pending(request: Request) -> UiFeedbackPendingResponse:
    container: ServiceContainer = request.app.state.container
    return UiFeedbackPendingResponse(pending=feedback_pending_count(container))


def _store_feedback_example(
    container: ServiceContainer,
    *,
    request_id: str,
    rating: str,
    detail: str | None,
    corrected_backend: str | None,
) -> None:
    """Closed feedback loop: snapshot (prompt, context, response) for
    thumbs-downs and trigger gated retraining at the configured threshold.

    Data integrity: feedback for a request_id with no recorded metric stores
    nothing (there is nothing to learn from), and repeat feedback for the
    same request_id replaces the earlier example (latest verdict wins), so
    the retrain threshold counts distinct requests only.

    Privacy: requests flagged sensitive at routing time (private-mode reroute
    or learned sensitivity escalation) never get a context snapshot, so their
    examples store context_text="". The prompt itself is still stored on an
    explicit thumbs-down because a "wrong model" correction can only train
    the router from the (prompt, corrected label) pair — submitting that
    correction is the user's deliberate choice.
    """
    preferences = container.personal_config.preferences
    try:
        from switchboard.app.models.telemetry import FeedbackExampleRecord
        from switchboard.training.feedback_loop import (
            FeedbackExampleStore,
            maybe_trigger_retraining,
        )

        engine = container.memory_repository.engine
        store = FeedbackExampleStore(engine)
        if rating == "good" or not preferences.store_feedback_examples:
            store.delete_example(request_id)
            return
        metric = container.backend_metrics_repository.get(request_id)
        if metric is None:
            # Unknown request: storing an empty example would only pad the
            # retrain threshold with noise.
            store.delete_example(request_id)
            return
        backend = metric.backend
        route_type = str(metric.metadata.get("route_type") or "") or None
        sensitive = bool(
            metric.metadata.get("private_mode_rerouted")
            or metric.metadata.get("sensitivity_escalated")
        )
        prompt_text = ""
        response_text = ""
        session_id = str(metric.metadata.get("session_id") or "")
        if session_id:
            for message in container.context_store.list_messages(session_id):
                if str(message.metadata.get("request_id") or "") == request_id:
                    if message.role == "user":
                        prompt_text = message.content
                    elif message.role == "assistant":
                        response_text = message.content
        store.add_example(
            FeedbackExampleRecord(
                request_id=request_id,
                rating=rating,
                detail=detail,
                corrected_backend=corrected_backend if detail == "wrong_model" else None,
                prompt=prompt_text,
                context_text="" if sensitive else store.get_recent_context(request_id),
                response_text=response_text,
                route_type=route_type,
                backend=backend,
            )
        )
        if preferences.feedback_auto_retrain:
            maybe_trigger_retraining(
                engine=engine,
                threshold=preferences.feedback_retrain_threshold,
                weights_path=preferences.router_weights_path,
            )
    except Exception:  # feedback storage must never fail the click
        pass
