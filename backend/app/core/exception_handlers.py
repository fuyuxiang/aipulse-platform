from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.core.tracing import current_request_id, current_trace_id


def _payload(code: str, message: str, details: object | None = None) -> dict[str, object]:
    return {
        "code": code,
        "message": message,
        "data": details,
        "request_id": current_request_id(),
        "trace_id": current_trace_id(),
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_payload(exc.code.value, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_payload(ErrorCode.VALIDATION_ERROR.value, "validation failed", exc.errors()))

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=_payload(ErrorCode.INTERNAL_ERROR.value, str(exc)))

