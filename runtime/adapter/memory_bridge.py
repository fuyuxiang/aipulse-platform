from __future__ import annotations

from typing import Any


class MemoryBridge:
    def apply_policy(self, agent_loop: Any, policy: dict[str, Any]) -> None:
        if policy.get("disabled") and hasattr(agent_loop, "config"):
            agent_loop.config.memory.enabled = False

