from __future__ import annotations

import sys
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.services._shared.resource_service import ResourceService

project_root = settings.project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from runtime.adapter.echo_agent_adapter import EchoAgentRuntimeAdapter  # noqa: E402
from runtime.adapter.lifecycle import RuntimeContext  # noqa: E402


class RuntimeControlService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)
        self.adapter = get_runtime_adapter()

    async def create_instance(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._hydrate_runtime_payload(tenant_id, agent_id, payload)
        workspace = settings.resolved_data_dir / "runtime" / tenant_id / agent_id / payload.get("session_id", "default")
        context = RuntimeContext(
            tenant_id=tenant_id,
            agent_id=agent_id,
            version_id=str(payload.get("version_id") or "draft"),
            session_id=str(payload.get("session_id") or "default"),
            workspace=str(workspace),
            model_config=dict(payload.get("model_config") or {}),
            tool_policy=dict(payload.get("tool_policy") or {}),
            memory_policy=dict(payload.get("memory_policy") or {}),
            knowledge_bindings=list(payload.get("knowledge_bindings") or []),
            resource_limits=dict(payload.get("resource_limits") or {}),
        )
        instance = await self.adapter.create(context)
        self.resources.create(
            "agent_runtime_instances",
            tenant_id,
            user_id,
            {
                "name": f"runtime {agent_id}",
                "status": instance.status,
                "agent_id": agent_id,
                "session_id": context.session_id,
                "version": context.version_id,
                "resource_type": "echo-agent",
                "spec": payload,
                "output_payload": {"instance_id": instance.id},
            },
        )
        return self._instance(instance)

    async def start(self, tenant_id: str, user_id: str, instance_id: str) -> dict[str, Any]:
        instance = await self.adapter.start(instance_id)
        self.resources.action("agent_runtime_instances", tenant_id, user_id, action="start", payload={"instance_id": instance_id})
        return self._instance(instance)

    async def stop(self, tenant_id: str, user_id: str, instance_id: str) -> dict[str, Any]:
        instance = await self.adapter.stop(instance_id)
        self.resources.action("agent_runtime_instances", tenant_id, user_id, action="stop", payload={"instance_id": instance_id})
        return self._instance(instance)

    async def restart(self, tenant_id: str, user_id: str, instance_id: str) -> dict[str, Any]:
        instance = await self.adapter.restart(instance_id)
        self.resources.action("agent_runtime_instances", tenant_id, user_id, action="restart", payload={"instance_id": instance_id})
        return self._instance(instance)

    async def destroy(self, tenant_id: str, user_id: str, instance_id: str) -> dict[str, Any]:
        instance = await self.adapter.destroy(instance_id)
        self.resources.action("agent_runtime_instances", tenant_id, user_id, action="destroy", payload={"instance_id": instance_id})
        return self._instance(instance)

    async def debug_run(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._hydrate_runtime_payload(tenant_id, agent_id, payload)
        instances = [item for item in self.adapter.list(tenant_id) if item.context.agent_id == agent_id]
        instance = instances[0] if instances else await self.adapter.create(
            RuntimeContext(
                tenant_id=tenant_id,
                agent_id=agent_id,
                version_id=str(payload.get("version_id") or "draft"),
                session_id=str(payload.get("session_id") or "debug"),
                workspace=str(settings.resolved_data_dir / "runtime" / tenant_id / agent_id / "debug"),
                model_config=dict(payload.get("model_config") or {}),
                tool_policy=dict(payload.get("tool_policy") or {}),
                memory_policy=dict(payload.get("memory_policy") or {}),
                knowledge_bindings=list(payload.get("knowledge_bindings") or []),
                resource_limits=dict(payload.get("resource_limits") or {}),
            )
        )
        result = await self.adapter.debug_run(instance.id, str(payload.get("prompt") or ""), session_id=str(payload.get("session_id") or "debug"))
        run = self.resources.create(
            "agent_run_records",
            tenant_id,
            user_id,
            {"name": "debug run", "status": "success", "agent_id": agent_id, "session_id": str(payload.get("session_id") or "debug"), "input_payload": payload, "output_payload": result},
        )
        return {"run_id": run.id, **result}

    def _hydrate_runtime_payload(self, tenant_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(payload)
        try:
            agent = self.resources.get("agents", tenant_id, agent_id)
        except Exception:
            return hydrated
        agent_config = dict(agent.config or {})
        agent_spec = dict(agent.spec or {})
        if not hydrated.get("model_config"):
            model_config = dict(agent_config.get("model_config") or agent_spec.get("model_config") or {})
            model_id = str(hydrated.get("model_id") or agent_config.get("model_id") or agent_spec.get("model_id") or "")
            if model_id:
                model_config = {**self._model_runtime_config(tenant_id, model_id), **model_config}
            hydrated["model_config"] = model_config
        if not hydrated.get("tool_policy"):
            hydrated["tool_policy"] = dict(agent_config.get("tool_policy") or agent_spec.get("tool_policy") or {})
        if not hydrated.get("memory_policy"):
            hydrated["memory_policy"] = dict(agent_config.get("memory_policy") or agent_spec.get("memory_policy") or {})
        if not hydrated.get("knowledge_bindings"):
            knowledge_ids = list(agent_config.get("knowledge_base_ids") or agent_spec.get("knowledge_base_ids") or [])
            hydrated["knowledge_bindings"] = [{"knowledge_base_id": item} for item in knowledge_ids]
        if not hydrated.get("resource_limits"):
            hydrated["resource_limits"] = dict(agent_config.get("resource_limits") or agent_spec.get("resource_limits") or {})
        return hydrated

    def _model_runtime_config(self, tenant_id: str, model_id: str) -> dict[str, Any]:
        model = self.resources.get("models", tenant_id, model_id)
        config = dict(model.config or {})
        provider_type = str(model.provider_type or "")
        if model.provider_id:
            provider = self.resources.get("model_providers", tenant_id, model.provider_id)
            config = {**dict(provider.config or {}), **config}
            provider_type = provider_type or str(provider.provider_type or "")
            endpoints, _ = self.resources.list("model_endpoints", tenant_id, 1, 50, {"provider_id": model.provider_id})
            for endpoint in endpoints:
                if endpoint.status == "active" and endpoint.enabled:
                    config = {**config, **dict(endpoint.config or {}), **dict(endpoint.spec or {})}
                    break
            credentials, _ = self.resources.list("model_credentials", tenant_id, 1, 50, {"provider_id": model.provider_id})
            for credential in credentials:
                if credential.status == "active" and credential.enabled:
                    config = {**config, **dict(credential.config or {}), **dict(credential.spec or {})}
                    break
        config["provider_type"] = provider_type or config.get("provider_type") or "echo_agent_native"
        config["model_name"] = model.model_id or config.get("model_name") or model.code or model.name
        config.setdefault("provider_name", config["provider_type"])
        return config

    def list_instances(self, tenant_id: str) -> list[dict[str, Any]]:
        return [self._instance(item) for item in self.adapter.list(tenant_id)]

    @staticmethod
    def _instance(instance: Any) -> dict[str, Any]:
        return {
            "id": instance.id,
            "status": instance.status,
            "tenant_id": instance.context.tenant_id,
            "agent_id": instance.context.agent_id,
            "version_id": instance.context.version_id,
            "session_id": instance.context.session_id,
            "workspace": instance.context.workspace,
        }


_adapter: EchoAgentRuntimeAdapter | None = None


def get_runtime_adapter() -> EchoAgentRuntimeAdapter:
    global _adapter
    if _adapter is None:
        _adapter = EchoAgentRuntimeAdapter(settings.resolved_echo_agent_path, settings.resolved_data_dir)
    return _adapter
