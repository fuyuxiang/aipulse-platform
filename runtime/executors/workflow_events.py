"""Workflow events — event-driven execution support.

Provides event subscription, waiting, triggering, and webhook capabilities
for workflow nodes that need to pause and wait for external signals.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class EventStatus(str, Enum):
    WAITING = "waiting"
    RECEIVED = "received"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class EventSubscription:
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    workflow_run_id: str = ""
    node_id: str = ""
    event_name: str = ""
    event_filter: dict[str, Any] = field(default_factory=dict)
    status: EventStatus = EventStatus.WAITING
    timeout_seconds: float = 3600.0
    created_at: float = field(default_factory=time.time)
    received_at: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.timeout_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.timeout_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_run_id": self.workflow_run_id,
            "node_id": self.node_id,
            "event_name": self.event_name,
            "event_filter": self.event_filter,
            "status": self.status.value,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "received_at": self.received_at,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventSubscription:
        return cls(
            id=data.get("id", ""),
            workflow_run_id=data.get("workflow_run_id", ""),
            node_id=data.get("node_id", ""),
            event_name=data.get("event_name", ""),
            event_filter=data.get("event_filter", {}),
            status=EventStatus(data.get("status", "waiting")),
            timeout_seconds=data.get("timeout_seconds", 3600.0),
            created_at=data.get("created_at", 0.0),
            received_at=data.get("received_at", 0.0),
            payload=data.get("payload", {}),
        )


@dataclass
class EmittedEvent:
    id: str = field(default_factory=lambda: f"em_{uuid.uuid4().hex[:12]}")
    event_name: str = ""
    source_workflow_run_id: str = ""
    source_node_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    emitted_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_name": self.event_name,
            "source_workflow_run_id": self.source_workflow_run_id,
            "source_node_id": self.source_node_id,
            "payload": self.payload,
            "emitted_at": self.emitted_at,
        }


class EventBus:
    """In-process event bus for workflow event coordination.

    Manages subscriptions and delivers events to waiting workflows.
    For distributed deployments, this should be backed by a message broker.
    """

    def __init__(self):
        self._subscriptions: dict[str, EventSubscription] = {}
        self._waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._emitted: list[EmittedEvent] = []

    def subscribe(
        self,
        workflow_run_id: str,
        node_id: str,
        event_name: str,
        event_filter: dict[str, Any] | None = None,
        timeout_seconds: float = 3600.0,
    ) -> EventSubscription:
        """Create a subscription for an event."""
        sub = EventSubscription(
            workflow_run_id=workflow_run_id,
            node_id=node_id,
            event_name=event_name,
            event_filter=event_filter or {},
            timeout_seconds=timeout_seconds,
        )
        self._subscriptions[sub.id] = sub
        return sub

    async def wait_for_event(
        self,
        subscription_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Wait for an event to be delivered to a subscription."""
        sub = self._subscriptions.get(subscription_id)
        if not sub:
            raise ValueError(f"subscription {subscription_id} not found")
        if sub.status == EventStatus.RECEIVED:
            return sub.payload

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._waiters[subscription_id] = future

        effective_timeout = timeout or sub.timeout_seconds or None
        try:
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            return result
        except asyncio.TimeoutError:
            sub.status = EventStatus.EXPIRED
            self._waiters.pop(subscription_id, None)
            raise
        finally:
            self._waiters.pop(subscription_id, None)

    def trigger(
        self,
        event_name: str,
        payload: dict[str, Any] | None = None,
        source_run_id: str = "",
        source_node_id: str = "",
    ) -> list[str]:
        """Trigger an event, delivering it to all matching subscriptions.

        Returns list of subscription IDs that were notified.
        """
        emitted = EmittedEvent(
            event_name=event_name,
            source_workflow_run_id=source_run_id,
            source_node_id=source_node_id,
            payload=payload or {},
        )
        self._emitted.append(emitted)

        notified: list[str] = []
        for sub_id, sub in list(self._subscriptions.items()):
            if sub.status != EventStatus.WAITING:
                continue
            if sub.event_name != event_name:
                continue
            if sub.is_expired:
                sub.status = EventStatus.EXPIRED
                continue
            if not self._filter_matches(sub.event_filter, payload or {}):
                continue

            sub.status = EventStatus.RECEIVED
            sub.received_at = time.time()
            sub.payload = payload or {}
            notified.append(sub_id)

            waiter = self._waiters.pop(sub_id, None)
            if waiter and not waiter.done():
                waiter.set_result(payload or {})

        return notified

    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a pending subscription."""
        sub = self._subscriptions.get(subscription_id)
        if not sub or sub.status != EventStatus.WAITING:
            return False
        sub.status = EventStatus.CANCELLED
        waiter = self._waiters.pop(subscription_id, None)
        if waiter and not waiter.done():
            waiter.set_exception(asyncio.CancelledError())
        return True

    def get_subscription(self, subscription_id: str) -> EventSubscription | None:
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(
        self,
        workflow_run_id: str = "",
        status: EventStatus | None = None,
    ) -> list[EventSubscription]:
        results = []
        for sub in self._subscriptions.values():
            if workflow_run_id and sub.workflow_run_id != workflow_run_id:
                continue
            if status and sub.status != status:
                continue
            results.append(sub)
        return results

    def list_emitted(self, event_name: str = "") -> list[EmittedEvent]:
        if not event_name:
            return list(self._emitted)
        return [e for e in self._emitted if e.event_name == event_name]

    def cleanup_expired(self) -> int:
        """Remove expired subscriptions. Returns count removed."""
        expired = [sid for sid, sub in self._subscriptions.items() if sub.is_expired and sub.status == EventStatus.WAITING]
        for sid in expired:
            self._subscriptions[sid].status = EventStatus.EXPIRED
            waiter = self._waiters.pop(sid, None)
            if waiter and not waiter.done():
                waiter.set_exception(asyncio.TimeoutError())
        return len(expired)

    @staticmethod
    def _filter_matches(event_filter: dict[str, Any], payload: dict[str, Any]) -> bool:
        """Check if payload matches the subscription filter."""
        if not event_filter:
            return True
        for key, expected in event_filter.items():
            actual = payload.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


_global_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus singleton."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_event_bus
    _global_event_bus = None
