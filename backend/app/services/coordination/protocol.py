"""Structured communication protocol for multi-agent coordination.

Defines standard message formats, correlation tracking, acknowledgment
patterns, and an in-process message broker for agent-to-agent communication.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class MessageType(str, Enum):
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    TASK_PROGRESS = "task_progress"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    CAPABILITY_ANNOUNCE = "capability_announce"
    CONFLICT_DETECTED = "conflict_detected"
    ESCALATION = "escalation"
    ACK = "ack"
    NACK = "nack"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class CoordinationMessage:
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    correlation_id: str = ""
    sender_agent_id: str = ""
    receiver_agent_id: str = ""
    message_type: MessageType = MessageType.STATUS_UPDATE
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0
    requires_ack: bool = False
    priority: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds

    @property
    def is_broadcast(self) -> bool:
        return not self.receiver_agent_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "sender_agent_id": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "requires_ack": self.requires_ack,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoordinationMessage:
        return cls(
            id=data.get("id", ""),
            correlation_id=data.get("correlation_id", ""),
            sender_agent_id=data.get("sender_agent_id", ""),
            receiver_agent_id=data.get("receiver_agent_id", ""),
            message_type=MessageType(data.get("message_type", "status_update")),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", 0.0),
            ttl_seconds=data.get("ttl_seconds", 300.0),
            requires_ack=data.get("requires_ack", False),
            priority=data.get("priority", 5),
            metadata=data.get("metadata", {}),
        )

    def create_reply(self, message_type: MessageType, payload: dict[str, Any], sender_id: str = "") -> CoordinationMessage:
        return CoordinationMessage(
            correlation_id=self.correlation_id or self.id,
            sender_agent_id=sender_id or self.receiver_agent_id,
            receiver_agent_id=self.sender_agent_id,
            message_type=message_type,
            payload=payload,
        )

    def create_ack(self, sender_id: str = "") -> CoordinationMessage:
        return self.create_reply(MessageType.ACK, {"acked_message_id": self.id}, sender_id)


MessageHandler = Callable[[CoordinationMessage], Awaitable[None]]


class MessageBroker:
    """In-process message broker for agent coordination.

    Routes messages between agents, handles subscriptions, tracks delivery,
    and manages acknowledgment timeouts.
    """

    def __init__(self, ack_timeout_seconds: float = 30.0):
        self._handlers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._broadcast_handlers: list[MessageHandler] = []
        self._pending_acks: dict[str, asyncio.Future[CoordinationMessage]] = {}
        self._message_log: list[CoordinationMessage] = []
        self._delivery_status: dict[str, DeliveryStatus] = {}
        self._ack_timeout = ack_timeout_seconds

    def subscribe(self, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to receive directed messages."""
        self._handlers[agent_id].append(handler)

    def subscribe_broadcast(self, handler: MessageHandler) -> None:
        """Subscribe to all broadcast messages."""
        self._broadcast_handlers.append(handler)

    def unsubscribe(self, agent_id: str) -> None:
        """Remove all handlers for an agent."""
        self._handlers.pop(agent_id, None)

    async def send(self, message: CoordinationMessage) -> DeliveryStatus:
        """Send a message to its target agent(s)."""
        if message.is_expired:
            self._delivery_status[message.id] = DeliveryStatus.EXPIRED
            return DeliveryStatus.EXPIRED

        self._message_log.append(message)

        if message.message_type == MessageType.ACK:
            self._handle_ack(message)
            self._delivery_status[message.id] = DeliveryStatus.DELIVERED
            return DeliveryStatus.DELIVERED

        if message.is_broadcast:
            await self._deliver_broadcast(message)
        else:
            await self._deliver_direct(message)

        if message.requires_ack:
            self._delivery_status[message.id] = DeliveryStatus.PENDING
        else:
            self._delivery_status[message.id] = DeliveryStatus.DELIVERED

        return self._delivery_status[message.id]

    async def send_and_wait_ack(self, message: CoordinationMessage, timeout: float | None = None) -> CoordinationMessage | None:
        """Send a message and wait for acknowledgment."""
        message.requires_ack = True
        future: asyncio.Future[CoordinationMessage] = asyncio.get_event_loop().create_future()
        self._pending_acks[message.id] = future

        await self.send(message)

        effective_timeout = timeout or self._ack_timeout
        try:
            ack = await asyncio.wait_for(future, timeout=effective_timeout)
            self._delivery_status[message.id] = DeliveryStatus.ACKNOWLEDGED
            return ack
        except asyncio.TimeoutError:
            self._delivery_status[message.id] = DeliveryStatus.EXPIRED
            self._pending_acks.pop(message.id, None)
            return None

    async def request_response(
        self,
        message: CoordinationMessage,
        response_type: MessageType | None = None,
        timeout: float = 30.0,
    ) -> CoordinationMessage | None:
        """Send a message and wait for a correlated response."""
        correlation_id = message.correlation_id or message.id
        message.correlation_id = correlation_id

        future: asyncio.Future[CoordinationMessage] = asyncio.get_event_loop().create_future()
        self._pending_acks[correlation_id] = future

        await self.send(message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_acks.pop(correlation_id, None)
            return None

    def get_delivery_status(self, message_id: str) -> DeliveryStatus:
        return self._delivery_status.get(message_id, DeliveryStatus.PENDING)

    def get_message_history(
        self,
        agent_id: str = "",
        message_type: MessageType | None = None,
        limit: int = 100,
    ) -> list[CoordinationMessage]:
        results = []
        for msg in reversed(self._message_log):
            if agent_id and msg.sender_agent_id != agent_id and msg.receiver_agent_id != agent_id:
                continue
            if message_type and msg.message_type != message_type:
                continue
            results.append(msg)
            if len(results) >= limit:
                break
        return results

    async def _deliver_direct(self, message: CoordinationMessage) -> None:
        handlers = self._handlers.get(message.receiver_agent_id, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception:
                pass

    async def _deliver_broadcast(self, message: CoordinationMessage) -> None:
        all_handlers = list(self._broadcast_handlers)
        for agent_id, handlers in self._handlers.items():
            if agent_id != message.sender_agent_id:
                all_handlers.extend(handlers)
        for handler in all_handlers:
            try:
                await handler(message)
            except Exception:
                pass

    def _handle_ack(self, ack_message: CoordinationMessage) -> None:
        acked_id = ack_message.payload.get("acked_message_id", "")
        correlation = ack_message.correlation_id

        for key in (acked_id, correlation):
            if key and key in self._pending_acks:
                future = self._pending_acks.pop(key)
                if not future.done():
                    future.set_result(ack_message)
                return

        if ack_message.message_type in (MessageType.TASK_RESULT, MessageType.VOTE_RESPONSE):
            if correlation and correlation in self._pending_acks:
                future = self._pending_acks.pop(correlation)
                if not future.done():
                    future.set_result(ack_message)

    def cleanup_expired(self) -> int:
        """Clean up expired pending acks."""
        expired = []
        for msg_id, future in list(self._pending_acks.items()):
            if future.done():
                expired.append(msg_id)
        for msg_id in expired:
            self._pending_acks.pop(msg_id, None)
        return len(expired)
