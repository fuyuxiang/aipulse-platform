from __future__ import annotations

from typing import Any


class ToolPolicyBridge:
    def allowed_tools(self, policy: dict[str, Any]) -> list[str]:
        return list(policy.get("allow") or [])

    def apply(self, agent_loop: Any, policy: dict[str, Any]) -> None:
        deny = set(policy.get("deny") or [])
        for name in list(agent_loop.tools.tool_names):
            if name in deny:
                agent_loop.tools.unregister(name)

