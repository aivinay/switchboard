from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from switchboard.app.models.catalogue import ModelProfile
from switchboard.app.models.telemetry import TelemetryRead
from switchboard.app.services.container import ServiceContainer

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/models", response_model=list[ModelProfile])
async def list_models(request: Request) -> list[ModelProfile]:
    container: ServiceContainer = request.app.state.container
    return container.catalogue.models


@router.get("/requests", response_model=list[TelemetryRead])
async def list_requests(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[TelemetryRead]:
    container: ServiceContainer = request.app.state.container
    return container.telemetry.repository.list(limit=limit)


@router.get("/requests/{request_id}", response_model=TelemetryRead)
async def get_request(request_id: str, request: Request) -> TelemetryRead:
    container: ServiceContainer = request.app.state.container
    record = container.telemetry.repository.get(request_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="request not found")
    return record


@router.get("/metrics/summary")
async def metrics_summary(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    return container.telemetry.repository.summary()


@router.get("/metrics/savings")
async def metrics_savings(request: Request) -> dict[str, object]:
    container: ServiceContainer = request.app.state.container
    summary = container.telemetry.repository.summary()
    baseline_model = container.catalogue.frontier_baseline()
    return {
        "baseline": "everything_goes_to_frontier_model",
        "baseline_model_id": baseline_model.model_id,
        "estimated_total_cost_usd": summary["estimated_total_cost_usd"],
        "estimated_baseline_cost_usd": summary["estimated_baseline_cost_usd"],
        "estimated_savings_usd": summary["estimated_savings_usd"],
        "total_requests": summary["total_requests"],
    }
