from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class ApprovalDecision:
    approval_id: str
    approved: bool
    decided_by: str
    reason: str = ""


class ApprovalWaiter:
    def __init__(self) -> None:
        self._events: dict[str, asyncio.Future[ApprovalDecision]] = {}

    def create(self, approval_id: str) -> None:
        self._events[approval_id] = asyncio.get_running_loop().create_future()

    async def wait(self, approval_id: str, timeout_seconds: float = 300) -> ApprovalDecision:
        future = self._events.setdefault(approval_id, asyncio.get_running_loop().create_future())
        return await asyncio.wait_for(future, timeout=timeout_seconds)

    def decide(self, decision: ApprovalDecision) -> None:
        future = self._events.setdefault(decision.approval_id, asyncio.get_running_loop().create_future())
        if not future.done():
            future.set_result(decision)
