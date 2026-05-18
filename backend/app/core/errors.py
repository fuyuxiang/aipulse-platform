from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.constants import ErrorCode


@dataclass(slots=True)
class AppError(Exception):
    code: ErrorCode
    message: str
    status_code: int = 400
    details: dict[str, Any] | None = None

