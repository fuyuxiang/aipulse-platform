from __future__ import annotations

from typing import Any


class KnowledgeBridge:
    def bind(self, agent_loop: Any, bindings: list[dict[str, Any]]) -> None:
        if bindings and hasattr(agent_loop, "context"):
            agent_loop.context.extra_context = {"knowledge_bindings": bindings}

