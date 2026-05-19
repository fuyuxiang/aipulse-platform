"""Health monitoring and fault tolerance for multi-agent teams.

Tracks agent heartbeats, detects failures, and manages automatic failover
by reassigning tasks from unhealthy agents to healthy alternatives.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentHealth:
    agent_id: str
    status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    last_heartbeat: float = 0.0
    consecutive_failures: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    avg_response_time_ms: float = 0.0
    current_load: int = 0
    max_capacity: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.total_tasks_completed + self.total_tasks_failed
        if total == 0:
            return 1.0
        return self.total_tasks_completed / total

    @property
    def is_available(self) -> bool:
        return self.status in (AgentHealthStatus.HEALTHY, AgentHealthStatus.DEGRADED) and self.current_load < self.max_capacity

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "consecutive_failures": self.consecutive_failures,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "avg_response_time_ms": self.avg_response_time_ms,
            "current_load": self.current_load,
            "max_capacity": self.max_capacity,
            "success_rate": self.success_rate,
            "is_available": self.is_available,
        }


@dataclass
class FailoverEvent:
    id: str = field(default_factory=lambda: f"fo_{uuid.uuid4().hex[:12]}")
    failed_agent_id: str = ""
    replacement_agent_id: str = ""
    task_description: str = ""
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "failed_agent_id": self.failed_agent_id,
            "replacement_agent_id": self.replacement_agent_id,
            "task_description": self.task_description,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


FailoverCallback = Callable[[FailoverEvent], Awaitable[None]]


class HealthMonitor:
    """Monitors agent health via heartbeats and manages failover.

    Configuration:
        heartbeat_interval: Expected interval between heartbeats (seconds)
        unhealthy_threshold: Number of missed heartbeats before marking unhealthy
        degraded_threshold: Number of consecutive failures before marking degraded
        check_interval: How often to run health checks (seconds)
    """

    def __init__(
        self,
        *,
        heartbeat_interval: float = 30.0,
        unhealthy_threshold: int = 3,
        degraded_threshold: int = 2,
        failover_callback: FailoverCallback | None = None,
    ):
        self._heartbeat_interval = heartbeat_interval
        self._unhealthy_threshold = unhealthy_threshold
        self._degraded_threshold = degraded_threshold
        self._failover_callback = failover_callback
        self._agents: dict[str, AgentHealth] = {}
        self._failover_history: list[FailoverEvent] = []
        self._pending_tasks: dict[str, list[dict[str, Any]]] = {}

    def register_agent(
        self,
        agent_id: str,
        max_capacity: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> AgentHealth:
        """Register an agent for health monitoring."""
        health = AgentHealth(
            agent_id=agent_id,
            status=AgentHealthStatus.HEALTHY,
            last_heartbeat=time.time(),
            max_capacity=max_capacity,
            metadata=metadata or {},
        )
        self._agents[agent_id] = health
        self._pending_tasks.setdefault(agent_id, [])
        return health

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from monitoring."""
        self._agents.pop(agent_id, None)
        self._pending_tasks.pop(agent_id, None)

    def record_heartbeat(self, agent_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a heartbeat from an agent."""
        health = self._agents.get(agent_id)
        if not health:
            health = self.register_agent(agent_id)
        health.last_heartbeat = time.time()
        if health.status == AgentHealthStatus.UNHEALTHY:
            health.status = AgentHealthStatus.DEGRADED
            health.consecutive_failures = 0
        elif health.status in (AgentHealthStatus.DEGRADED, AgentHealthStatus.UNKNOWN):
            health.status = AgentHealthStatus.HEALTHY
            health.consecutive_failures = 0
        if metadata:
            health.metadata.update(metadata)
            if "current_load" in metadata:
                health.current_load = int(metadata["current_load"])

    def record_task_success(self, agent_id: str, response_time_ms: float = 0.0) -> None:
        """Record a successful task completion."""
        health = self._agents.get(agent_id)
        if not health:
            return
        health.total_tasks_completed += 1
        health.consecutive_failures = 0
        if response_time_ms > 0:
            total = health.total_tasks_completed + health.total_tasks_failed
            health.avg_response_time_ms = (
                (health.avg_response_time_ms * (total - 1) + response_time_ms) / total
            )
        if health.status == AgentHealthStatus.DEGRADED:
            health.status = AgentHealthStatus.HEALTHY

    def record_task_failure(self, agent_id: str, error: str = "") -> None:
        """Record a task failure."""
        health = self._agents.get(agent_id)
        if not health:
            return
        health.total_tasks_failed += 1
        health.consecutive_failures += 1
        if health.consecutive_failures >= self._degraded_threshold:
            if health.status == AgentHealthStatus.HEALTHY:
                health.status = AgentHealthStatus.DEGRADED

    def assign_task(self, agent_id: str, task: dict[str, Any]) -> None:
        """Track a task assigned to an agent."""
        health = self._agents.get(agent_id)
        if health:
            health.current_load += 1
        self._pending_tasks.setdefault(agent_id, []).append(task)

    def complete_task(self, agent_id: str, task_id: str = "") -> None:
        """Mark a task as completed, reducing agent load."""
        health = self._agents.get(agent_id)
        if health and health.current_load > 0:
            health.current_load -= 1
        tasks = self._pending_tasks.get(agent_id, [])
        if task_id:
            self._pending_tasks[agent_id] = [t for t in tasks if t.get("id") != task_id]
        elif tasks:
            self._pending_tasks[agent_id] = tasks[:-1]

    async def check_health(self) -> list[FailoverEvent]:
        """Run health check on all agents. Returns failover events if any."""
        now = time.time()
        failovers: list[FailoverEvent] = []

        for agent_id, health in list(self._agents.items()):
            if health.status in (AgentHealthStatus.HEALTHY, AgentHealthStatus.DEGRADED):
                missed = (now - health.last_heartbeat) / self._heartbeat_interval
                if missed >= self._unhealthy_threshold:
                    health.status = AgentHealthStatus.UNHEALTHY
                    pending = self._pending_tasks.get(agent_id, [])
                    if pending:
                        events = await self._failover_tasks(agent_id, pending)
                        failovers.extend(events)

        return failovers

    async def _failover_tasks(self, failed_agent_id: str, tasks: list[dict[str, Any]]) -> list[FailoverEvent]:
        """Reassign tasks from a failed agent to healthy alternatives."""
        events: list[FailoverEvent] = []
        healthy_agents = self.get_available_agents(exclude=[failed_agent_id])

        for task in tasks:
            replacement = self._select_replacement(healthy_agents, task)
            if not replacement:
                continue

            event = FailoverEvent(
                failed_agent_id=failed_agent_id,
                replacement_agent_id=replacement.agent_id,
                task_description=str(task.get("description", task.get("title", ""))),
                reason=f"agent {failed_agent_id} unhealthy (missed heartbeats)",
            )
            events.append(event)
            self._failover_history.append(event)

            replacement.current_load += 1
            self._pending_tasks.setdefault(replacement.agent_id, []).append(task)

            if self._failover_callback:
                try:
                    await self._failover_callback(event)
                except Exception:
                    pass

        self._pending_tasks[failed_agent_id] = []
        return events

    def _select_replacement(
        self,
        candidates: list[AgentHealth],
        task: dict[str, Any],
    ) -> AgentHealth | None:
        """Select the best replacement agent for a failed task."""
        if not candidates:
            return None

        scored = []
        for agent in candidates:
            score = (
                agent.success_rate * 0.4
                + (agent.max_capacity - agent.current_load) / max(1, agent.max_capacity) * 0.4
                + (1.0 - agent.consecutive_failures * 0.1) * 0.2
            )
            scored.append((agent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def get_agent_health(self, agent_id: str) -> AgentHealth | None:
        return self._agents.get(agent_id)

    def get_all_health(self) -> list[AgentHealth]:
        return list(self._agents.values())

    def get_available_agents(self, exclude: list[str] | None = None) -> list[AgentHealth]:
        """Get all agents that are available for work."""
        excluded = set(exclude or [])
        return [
            h for h in self._agents.values()
            if h.is_available and h.agent_id not in excluded
        ]

    def get_failover_history(self, limit: int = 50) -> list[FailoverEvent]:
        return self._failover_history[-limit:]

    def get_team_status(self) -> dict[str, Any]:
        """Get overall team health summary."""
        agents = list(self._agents.values())
        return {
            "total_agents": len(agents),
            "healthy": sum(1 for a in agents if a.status == AgentHealthStatus.HEALTHY),
            "degraded": sum(1 for a in agents if a.status == AgentHealthStatus.DEGRADED),
            "unhealthy": sum(1 for a in agents if a.status == AgentHealthStatus.UNHEALTHY),
            "unknown": sum(1 for a in agents if a.status == AgentHealthStatus.UNKNOWN),
            "total_capacity": sum(a.max_capacity for a in agents),
            "current_load": sum(a.current_load for a in agents),
            "available_capacity": sum(a.max_capacity - a.current_load for a in agents if a.is_available),
            "failovers_total": len(self._failover_history),
        }
