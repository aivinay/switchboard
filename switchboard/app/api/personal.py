from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request

from switchboard.app.models.personal import (
    FeedbackCreate,
    FeedbackRead,
    PersonalAskResponse,
    PersonalMemoryCreate,
    PersonalMemoryRead,
    PersonalModelRead,
    PersonalPromptRequest,
    PersonalRouteResponse,
)
from switchboard.app.models.telemetry import PersonalTelemetryRead
from switchboard.app.services.container import ServiceContainer
from switchboard.app.services.personal_switchboard import (
    PersonalRoutingError,
    PersonalSwitchboardService,
)

router = APIRouter(prefix="/personal", tags=["personal"])


def personal_service(request: Request) -> PersonalSwitchboardService:
    container: ServiceContainer = request.app.state.container
    return PersonalSwitchboardService(container)


@router.get("/health")
async def personal_health(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    return {
        "status": "ok",
        "product": "Switchboard",
        "environment": container.settings.environment,
        "user_id": container.personal_config.profile.user_id,
        "default_project": container.personal_config.profile.default_project,
        "local_first": container.personal_config.preferences.local_first,
        "allow_cloud": container.personal_config.preferences.allow_cloud,
        "private_mode": container.personal_config.preferences.private_mode,
    }


@router.post("/route", response_model=PersonalRouteResponse)
async def route_prompt(payload: PersonalPromptRequest, request: Request) -> PersonalRouteResponse:
    try:
        return personal_service(request).route(payload)
    except PersonalRoutingError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "PERSONAL_ROUTING_ERROR", "message": str(exc)},
        ) from exc


@router.post("/ask", response_model=PersonalAskResponse)
async def ask_prompt(payload: PersonalPromptRequest, request: Request) -> PersonalAskResponse:
    try:
        return await personal_service(request).ask(payload)
    except PersonalRoutingError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "PERSONAL_ROUTING_ERROR", "message": str(exc)},
        ) from exc


@router.get("/models", response_model=list[PersonalModelRead])
async def list_personal_models(request: Request) -> list[PersonalModelRead]:
    return personal_service(request).models()


@router.get("/usage")
async def usage(request: Request) -> dict[str, object]:
    return personal_service(request).usage()


@router.get("/savings")
async def savings(
    request: Request,
    days: int = Query(default=7, ge=1, le=365),
    since: str | None = None,
) -> dict[str, object]:
    since_dt = datetime.fromisoformat(since) if since else None
    return personal_service(request).savings(days=None if since_dt else days, since=since_dt)


@router.get("/history", response_model=list[PersonalTelemetryRead])
async def history(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[PersonalTelemetryRead]:
    return personal_service(request).history(limit=limit)


@router.post("/memory", response_model=PersonalMemoryRead)
async def add_memory(payload: PersonalMemoryCreate, request: Request) -> PersonalMemoryRead:
    return personal_service(request).add_memory(payload)


@router.get("/memory/search", response_model=list[PersonalMemoryRead])
async def search_memory(
    request: Request,
    q: str,
    project: str | None = None,
) -> list[PersonalMemoryRead]:
    return personal_service(request).search_memory(q, project=project)


@router.post("/feedback", response_model=FeedbackRead)
async def add_feedback(payload: FeedbackCreate, request: Request) -> FeedbackRead:
    return personal_service(request).add_feedback(payload)
