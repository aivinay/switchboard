from __future__ import annotations

from fastapi import APIRouter, Request

from switchboard.app.services.container import ServiceContainer

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    return {
        "status": "ok",
        "product": "Switchboard",
        "environment": container.settings.environment,
        "enabled_models": len(container.catalogue.enabled_models()),
    }
