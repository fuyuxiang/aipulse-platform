from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from runtime.adapter.config_bridge import ConfigBridge
from runtime.adapter.knowledge_bridge import KnowledgeBridge
from runtime.adapter.lifecycle import RuntimeContext, RuntimeInstance
from runtime.adapter.memory_bridge import MemoryBridge
from runtime.adapter.model_bridge import ModelBridge
from runtime.adapter.tool_policy_bridge import ToolPolicyBridge


class InstanceManager:
    def __init__(self, echo_agent_path: Path, data_dir: Path):
        self.config_bridge = ConfigBridge(echo_agent_path)
        self.model_bridge = ModelBridge(self.config_bridge)
        self.data_dir = data_dir
        self.instances: dict[str, RuntimeInstance] = {}
        self._lock = asyncio.Lock()

    async def create(self, context: RuntimeContext) -> RuntimeInstance:
        async with self._lock:
            instance = RuntimeInstance(id=uuid.uuid4().hex, context=context)
            self.instances[instance.id] = instance
            return instance

    async def start(self, instance_id: str) -> RuntimeInstance:
        instance = self.get(instance_id)
        if instance.status == "running":
            return instance
        self.config_bridge.ensure_importable()
        from echo_agent.agent.loop import AgentLoop
        from echo_agent.bus.queue import MessageBus
        from echo_agent.models.router import ModelRouter
        from echo_agent.storage.sqlite import SQLiteBackend
        from echo_agent.tasks.manager import TaskManager
        from echo_agent.tasks.workflow import WorkflowEngine

        workspace = Path(instance.context.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        config = self.config_bridge.build_echo_config(workspace, instance.context.model_config, instance.context.tool_policy)
        storage = SQLiteBackend(workspace / "data" / "echo_agent.db")
        await storage.initialize()
        bus = MessageBus()
        provider = self.model_bridge.build_provider(instance.context.model_config)
        router = ModelRouter(config.models)
        router.register_provider("aipulse-local", provider)
        task_manager = TaskManager(storage)
        workflow_engine = WorkflowEngine(storage, task_manager)
        agent_loop = AgentLoop(bus=bus, config=config, provider=provider, workspace=workspace, router=router, storage=storage, task_manager=task_manager, workflow_engine=workflow_engine)
        MemoryBridge().apply_policy(agent_loop, instance.context.memory_policy)
        ToolPolicyBridge().apply(agent_loop, instance.context.tool_policy)
        KnowledgeBridge().bind(agent_loop, instance.context.knowledge_bindings)
        await bus.start()
        await agent_loop.start()
        instance.bus = bus
        instance.storage = storage
        instance.provider = provider
        instance.agent_loop = agent_loop
        instance.status = "running"
        return instance

    async def stop(self, instance_id: str) -> RuntimeInstance:
        instance = self.get(instance_id)
        if instance.agent_loop:
            await instance.agent_loop.stop()
        if instance.bus:
            await instance.bus.stop()
        if instance.storage:
            await instance.storage.close()
        instance.status = "stopped"
        return instance

    async def restart(self, instance_id: str) -> RuntimeInstance:
        await self.stop(instance_id)
        return await self.start(instance_id)

    async def destroy(self, instance_id: str) -> RuntimeInstance:
        instance = await self.stop(instance_id)
        self.instances.pop(instance_id, None)
        instance.status = "destroyed"
        return instance

    def get(self, instance_id: str) -> RuntimeInstance:
        if instance_id not in self.instances:
            raise KeyError(f"runtime instance not found: {instance_id}")
        return self.instances[instance_id]

    def list(self, tenant_id: str | None = None) -> list[RuntimeInstance]:
        rows = list(self.instances.values())
        return [row for row in rows if tenant_id is None or row.context.tenant_id == tenant_id]
