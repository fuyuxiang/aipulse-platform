from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    code: str = "OK"
    message: str = "ok"
    data: T | None = None
    request_id: str = ""
    trace_id: str = ""


class ListResponse(BaseModel, Generic[T]):
    items: list[T] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


def ok(data: Any = None, *, request_id: str = "", trace_id: str = "") -> APIResponse[Any]:
    return APIResponse(data=data, request_id=request_id, trace_id=trace_id)

