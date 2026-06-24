from __future__ import annotations

from fastapi import HTTPException, status

from switchboard.app.models.api import ErrorDetail


def http_error(
    status_code: int,
    code: str,
    message: str,
    reason_codes: list[str] | None = None,
) -> HTTPException:
    detail = ErrorDetail(
        code=code,
        message=message,
        reason_codes=reason_codes or [],
    ).model_dump()
    return HTTPException(status_code=status_code, detail=detail)


def streaming_not_implemented() -> HTTPException:
    return http_error(
        status.HTTP_400_BAD_REQUEST,
        "STREAMING_NOT_IMPLEMENTED",
        "Streaming chat completions are not implemented yet.",
        ["STREAMING_NOT_IMPLEMENTED"],
    )
