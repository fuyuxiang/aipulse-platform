from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TelemetryBridge:
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, tenant_id: str, run_id: str, event: dict[str, Any]) -> None:
        path = self.trace_dir / f"{tenant_id}.jsonl"
        payload = {"at": time.time(), "tenant_id": tenant_id, "run_id": run_id, **event}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

