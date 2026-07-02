from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from switchboard.app.models.backends import SwitchboardResponse, backend_display_name
from switchboard.app.models.personal import FeedbackCreate, FeedbackRead
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.core_factory import build_configured_core_service
from switchboard.app.services.personal_switchboard import PersonalSwitchboardService
from switchboard.app.services.quota import QuotaLedgerService
from switchboard.app.services.switchboard_core import SwitchboardCoreService

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


class UiChatRequest(BaseModel):
    message: str = Field(min_length=1)
    backend: str = "auto"
    session_id: str | None = None


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
    created_at: str


class UiHistoryResponse(BaseModel):
    session_id: str
    messages: list[UiHistoryMessage]


class UiFeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    rating: str = Field(min_length=1)
    note: str | None = None
    # Thumbs-down disambiguation: "bad_answer" or "wrong_model".
    detail: str | None = None
    corrected_backend: str | None = None  # ollama | codex | claude-code


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
        metadata={"surface": "ui", "requested_backend": selected_backend},
        session_id=payload.session_id,
    )
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


def stream_chat_response(response: SwitchboardResponse) -> Iterator[str]:
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
    routing_info = {
        "request_id": response.request_id,
        "routing_reason": response.routing_reason,
        "latency_ms": response.latency_ms,
        "cost_type": response.cost_type.value,
        "selected_model": response.selected_model,
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
    return StreamingResponse(
        stream_chat_response(response),
        media_type="application/x-ndjson",
    )


@router.get("/api/chat/history", response_model=UiHistoryResponse)
def chat_history(session_id: str, request: Request) -> UiHistoryResponse:
    container: ServiceContainer = request.app.state.container
    session = container.context_store.get_session(session_id)
    if session is None:
        return UiHistoryResponse(session_id=session_id, messages=[])
    records = container.context_store.list_messages(session_id)
    messages = [
        UiHistoryMessage(
            message_id=record.message_id,
            role=record.role,
            content=record.content,
            display_model=record.display_model,
            backend=record.backend,
            request_id=str(record.metadata.get("request_id") or "") or None,
            created_at=record.created_at.isoformat(),
        )
        for record in records
        if record.role in {"user", "assistant"}
    ]
    return UiHistoryResponse(session_id=session_id, messages=messages)


@router.get("/api/quota")
def quota_status(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    return QuotaLedgerService(
        container.backend_metrics_repository,
        container.personal_config.quota,
    ).snapshot()


VALID_CORRECTED_BACKENDS = ("ollama", "codex", "claude-code")


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
        )
    )
    _store_feedback_example(
        container,
        request_id=payload.request_id.strip(),
        rating=rating,
        detail=detail,
        corrected_backend=corrected,
    )
    return result


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
    if not preferences.store_feedback_examples or rating == "good":
        return
    try:
        from switchboard.app.models.telemetry import FeedbackExampleRecord
        from switchboard.training.feedback_loop import (
            FeedbackExampleStore,
            maybe_trigger_retraining,
        )

        engine = container.memory_repository.engine
        metric = container.backend_metrics_repository.get(request_id)
        if metric is None:
            # Unknown request: storing an empty example would only pad the
            # retrain threshold with noise.
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
        store = FeedbackExampleStore(engine)
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
        maybe_trigger_retraining(
            engine=engine,
            threshold=preferences.feedback_retrain_threshold,
            weights_path=preferences.router_weights_path,
        )
    except Exception:  # feedback storage must never fail the click
        pass
