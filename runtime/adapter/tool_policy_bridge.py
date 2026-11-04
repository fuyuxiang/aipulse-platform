from __future__ import annotations

from typing import Any


class ToolPolicyBridge:
    def allowed_tools(self, policy: dict[str, Any]) -> list[str]:
        return list(policy.get("allow") or [])

    def apply(self, agent_loop: Any, policy: dict[str, Any]) -> None:
        deny = set(policy.get("deny") or [])
        allowed = set(policy.get("allow") or [])
        for name in list(agent_loop.tools.tool_names):
            if name in deny or name not in allowed:
                agent_loop.tools.unregister(name)

