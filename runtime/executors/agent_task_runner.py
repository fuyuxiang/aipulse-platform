from __future__ import annotations

from typing import Any

from runtime.adapter.echo_agent_adapter import EchoAgentRuntimeAdapter


class AgentTaskRunner:
    def __init__(self, adapter: EchoAgentRuntimeAdapter):
        self.adapter = adapter

    async def run_debug(self, instance_id: str, prompt: str, session_id: str = "debug") -> dict[str, Any]:
        return await self.adapter.debug_run(instance_id, prompt, session_id=session_id)

