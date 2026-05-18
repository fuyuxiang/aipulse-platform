from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.adapter.instance_manager import InstanceManager
from runtime.adapter.lifecycle import RuntimeContext, RuntimeInstance
from runtime.adapter.resource_limits import ResourceLimits, run_with_limits
from runtime.adapter.session_bridge import SessionBridge


class EchoAgentRuntimeAdapter:
    def __init__(self, echo_agent_path: Path, data_dir: Path):
        self.manager = InstanceManager(echo_agent_path, data_dir)
        self.data_dir = data_dir

    async def create(self, context: RuntimeContext) -> RuntimeInstance:
        return await self.manager.create(context)

    async def start(self, instance_id: str) -> RuntimeInstance:
        return await self.manager.start(instance_id)

    async def stop(self, instance_id: str) -> RuntimeInstance:
        return await self.manager.stop(instance_id)

    async def restart(self, instance_id: str) -> RuntimeInstance:
        return await self.manager.restart(instance_id)

    async def destroy(self, instance_id: str) -> RuntimeInstance:
        return await self.manager.destroy(instance_id)

    def health_check(self, instance_id: str) -> dict[str, Any]:
        instance = self.manager.get(instance_id)
        return {"instance_id": instance_id, "status": instance.status, "running": instance.status == "running"}

    def list(self, tenant_id: str | None = None) -> list[RuntimeInstance]:
        return self.manager.list(tenant_id)

    async def debug_run(self, instance_id: str, prompt: str, session_id: str = "debug") -> dict[str, Any]:
        instance = self.manager.get(instance_id)
        if instance.status != "running":
            instance = await self.start(instance_id)
        bridge = SessionBridge(Path(instance.context.workspace))
        session_key = bridge.session_key(instance.context.tenant_id, instance.context.agent_id, instance.context.version_id, session_id)
        limits = ResourceLimits(**{k: v for k, v in instance.context.resource_limits.items() if k in {"timeout_seconds", "max_retries", "circuit_breaker_failures"}})

        async def operation() -> str:
            return await instance.agent_loop.process_direct(prompt, session_key=session_key)

        response = await run_with_limits(operation, limits)
        return {"instance_id": instance_id, "session_id": session_id, "response": response}

