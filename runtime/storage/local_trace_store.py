from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalTraceStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, trace_id: str, span: dict[str, Any]) -> None:
        with (self.root / f"{trace_id}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(span, ensure_ascii=False) + "\n")

    def read(self, trace_id: str) -> list[dict[str, Any]]:
        path = self.root / f"{trace_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

