"""Workflow compensation — Saga pattern implementation for rollback on failure.

When a workflow fails mid-execution, the compensation engine runs compensating
actions in reverse order for all previously completed nodes that defined
compensation handlers.

Supports two strategies:
- backward_recovery: Run compensations in reverse order (default saga pattern)
- forward_recovery: Retry the failed node before compensating
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

CompensationHandler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]]


class CompensationStrategy(str, Enum):
    BACKWARD = "backward_recovery"
    FORWARD = "forward_recovery"


class CompensationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CompensationRecord:
    id: str = field(default_factory=lambda: f"comp_{uuid.uuid4().hex[:12]}")
    node_id: str = ""
    status: CompensationStatus = CompensationStatus.PENDING
    compensation_config: dict[str, Any] = field(default_factory=dict)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "status": self.status.value,
            "compensation_config": self.compensation_config,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class CompensationResult:
    """Result of running the full compensation chain."""
    strategy: CompensationStrategy
    records: list[CompensationRecord] = field(default_factory=list)
    all_succeeded: bool = False
    failed_compensations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "all_succeeded": self.all_succeeded,
            "records": [r.to_dict() for r in self.records],
            "failed_compensations": self.failed_compensations,
        }


class CompensationEngine:
    """Executes compensation actions for failed workflows using the Saga pattern."""

    def __init__(
        self,
        handler: CompensationHandler,
        *,
        strategy: CompensationStrategy = CompensationStrategy.BACKWARD,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
        continue_on_error: bool = True,
    ):
        self._handler = handler
        self._strategy = strategy
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._continue_on_error = continue_on_error

    async def compensate(
        self,
        compensation_stack: list[dict[str, Any]],
        failed_node_id: str = "",
        retry_handler: CompensationHandler | None = None,
    ) -> CompensationResult:
        """Run compensation for all items in the stack.

        Args:
            compensation_stack: List of dicts with keys: node_id, compensation, context_snapshot
            failed_node_id: The node that triggered compensation
            retry_handler: Optional handler to retry the failed node (forward recovery)
        """
        result = CompensationResult(strategy=self._strategy)

        if self._strategy == CompensationStrategy.FORWARD and retry_handler and failed_node_id:
            retry_success = await self._attempt_forward_recovery(
                failed_node_id, compensation_stack, retry_handler
            )
            if retry_success:
                result.all_succeeded = True
                return result

        reversed_stack = list(reversed(compensation_stack))
        for item in reversed_stack:
            record = CompensationRecord(
                node_id=item.get("node_id", ""),
                compensation_config=item.get("compensation", {}),
                context_snapshot=item.get("context_snapshot", {}),
            )
            record.started_at = time.time()
            record.status = CompensationStatus.RUNNING

            success = await self._execute_with_retry(record)
            record.finished_at = time.time()

            if success:
                record.status = CompensationStatus.SUCCESS
            else:
                record.status = CompensationStatus.FAILED
                result.failed_compensations.append(record.node_id)
                if not self._continue_on_error:
                    result.records.append(record)
                    break

            result.records.append(record)

        result.all_succeeded = len(result.failed_compensations) == 0
        return result

    async def _execute_with_retry(self, record: CompensationRecord) -> bool:
        """Execute a single compensation action with retries."""
        for attempt in range(self._max_retries + 1):
            try:
                await self._handler(record.compensation_config, record.context_snapshot)
                return True
            except Exception as exc:
                record.error = f"attempt {attempt + 1}: {exc}"
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
        return False

    async def _attempt_forward_recovery(
        self,
        failed_node_id: str,
        compensation_stack: list[dict[str, Any]],
        retry_handler: CompensationHandler,
    ) -> bool:
        """Attempt to retry the failed node before compensating."""
        failed_item = next(
            (item for item in compensation_stack if item.get("node_id") == failed_node_id),
            None,
        )
        if not failed_item:
            return False
        try:
            await retry_handler(
                failed_item.get("compensation", {}),
                failed_item.get("context_snapshot", {}),
            )
            return True
        except Exception:
            return False


def build_compensation_config(
    node_type: str,
    node_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Build compensation configuration for a node based on its type.

    Returns None if the node type doesn't support compensation.
    """
    compensation = node_config.get("compensation")
    if compensation:
        return compensation

    if node_type == "http":
        rollback_url = node_config.get("rollback_url")
        if rollback_url:
            return {
                "type": "http",
                "method": node_config.get("rollback_method", "POST"),
                "url": rollback_url,
                "headers": node_config.get("headers", {}),
            }

    if node_type == "agent":
        rollback_prompt = node_config.get("rollback_prompt")
        if rollback_prompt:
            return {
                "type": "agent",
                "agent_id": node_config.get("agent_id", ""),
                "prompt": rollback_prompt,
            }

    if node_type == "tool":
        rollback_tool = node_config.get("rollback_tool")
        if rollback_tool:
            return {
                "type": "tool",
                "tool_id": rollback_tool,
                "arguments": node_config.get("rollback_arguments", {}),
            }

    return None
