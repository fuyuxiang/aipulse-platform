from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def new_id() -> str:
    return uuid.uuid4().hex


def current_request_id() -> str:
    return request_id_var.get()


def current_trace_id() -> str:
    return trace_id_var.get()


def write_trace(event: dict[str, Any]) -> None:
    path = settings.resolve_path(settings.trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "at": time.time(),
        "request_id": current_request_id(),
        "trace_id": current_trace_id(),
        "tenant_id": tenant_id_var.get(),
        "user_id": user_id_var.get(),
        **event,
    }
    try:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except PermissionError:
        return


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        request_id = request.headers.get("X-Request-ID") or new_id()
        trace_id = request.headers.get("X-Trace-ID") or new_id()
        request_id_var.set(request_id)
        trace_id_var.set(trace_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            write_trace({"name": "http.request", "status": "error", "path": request.url.path})
            raise
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        write_trace(
            {
                "name": "http.request",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
        )
        return response
