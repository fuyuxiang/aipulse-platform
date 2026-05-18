from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


@dataclass
class TaskRecord:
    id: str
    status: str = "queued"
    result: Any = None
    error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LocalTaskExecutor:
    def __init__(self, max_concurrency: int = 8):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.records: dict[str, TaskRecord] = {}
        self.idempotency: dict[str, str] = {}

    async def submit(self, operation: Callable[[], Awaitable[Any]], *, idempotency_key: str = "", timeout_seconds: float = 120, retries: int = 0) -> TaskRecord:
        if idempotency_key and idempotency_key in self.idempotency:
            return self.records[self.idempotency[idempotency_key]]
        record = TaskRecord(id=uuid.uuid4().hex)
        self.records[record.id] = record
        if idempotency_key:
            self.idempotency[idempotency_key] = record.id
        async with self.semaphore:
            last_error = ""
            for attempt in range(retries + 1):
                record.status = "running"
                record.updated_at = datetime.now(timezone.utc)
                try:
                    record.result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
                    record.status = "success"
                    record.updated_at = datetime.now(timezone.utc)
                    return record
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    record.error = last_error
                    record.status = "retrying" if attempt < retries else "failed"
            record.updated_at = datetime.now(timezone.utc)
            record.error = last_error
            return record

    def get(self, task_id: str) -> TaskRecord:
        return self.records[task_id]

