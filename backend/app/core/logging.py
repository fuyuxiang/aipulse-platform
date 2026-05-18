from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.tracing import current_request_id, current_trace_id, tenant_id_var, user_id_var


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": current_request_id(),
            "trace_id": current_trace_id(),
            "tenant_id": tenant_id_var.get(),
            "user_id": user_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    log_path = settings.resolve_path(settings.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except PermissionError:
        return
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
