from __future__ import annotations

from fastapi import APIRouter, Request

from switchboard.app.core.errors import streaming_not_implemented
from switchboard.app.models.api import ChatCompletionRequest, ChatCompletionResponse
from switchboard.app.services.chat_completion import ChatCompletionService
from switchboard.app.services.container import ServiceContainer

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    if payload.stream:
        raise streaming_not_implemented()

    container: ServiceContainer = request.app.state.container
    return await ChatCompletionService(container).complete(payload)
