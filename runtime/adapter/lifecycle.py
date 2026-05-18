from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class RuntimeContext:
    tenant_id: str
    agent_id: str
    version_id: str
    session_id: str
    workspace: str
    model_config: dict[str, Any] = field(default_factory=dict)
    tool_policy: dict[str, Any] = field(default_factory=dict)
    memory_policy: dict[str, Any] = field(default_factory=dict)
    knowledge_bindings: list[dict[str, Any]] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeInstance:
    id: str
    context: RuntimeContext
    status: str = "created"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    agent_loop: Any = None
    bus: Any = None
    storage: Any = None
    provider: Any = None

