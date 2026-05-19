from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.services.agent_runner_service import AgentRunnerService
from app.services.resource_service import ResourceService


class AgentService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def clone(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        suffix = datetime.now(timezone.utc).strftime("%H%M%S")
        clone = self.resources.create(
            "agents",
            tenant_id,
            user_id,
            {
                "name": str(payload.get("name") or f"{agent.name} Copy"),
                "code": str(payload.get("code") or f"{agent.code or agent.id}-copy-{suffix}"),
                "description": agent.description,
                "status": "draft",
                "enabled": agent.enabled,
                "resource_type": agent.resource_type,
                "owner_id": user_id,
                "version": "draft",
                "model_type": agent.model_type,
                "config": agent.config,
                "spec": {**(agent.spec or {}), "cloned_from": agent.id},
                "metadata": {"cloned_from": agent.id},
            },
        )
        return ResourceService.to_dict(clone)

    def create_version(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        version = str(payload.get("version") or agent.version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        version_row = self.resources.create(
            "agent_versions",
            tenant_id,
            user_id,
            {
                "name": f"{agent.name} {version}",
                "status": "draft",
                "parent_id": agent.id,
                "agent_id": agent.id,
                "version": version,
                "model_type": agent.model_type,
                "config": payload.get("config") or agent.config,
                "spec": {"agent": ResourceService.to_dict(agent), "change_summary": payload.get("change_summary", "")},
            },
        )
        return ResourceService.to_dict(version_row)

    def release(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        version_id = str(payload.get("version_id") or "")
        version = self.resources.get("agent_versions", tenant_id, version_id) if version_id else self._latest_version(tenant_id, user_id, agent.id)
        release = self.resources.create(
            "agent_releases",
            tenant_id,
            user_id,
            {
                "name": f"release {agent.name} {version.version}",
                "status": "released",
                "parent_id": agent.id,
                "agent_id": agent.id,
                "version": version.version,
                "config": {"version_id": version.id, "channel": payload.get("channel", "stable")},
                "spec": payload,
            },
        )
        self.resources.update("agents", tenant_id, user_id, agent.id, {"status": "released", "version": version.version, "enabled": True})
        return ResourceService.to_dict(release)

    def gray_release(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        percentage = int(payload.get("percentage") or payload.get("traffic_percentage") or 10)
        if percentage < 1 or percentage > 100:
            raise AppError(ErrorCode.VALIDATION_ERROR, "gray release percentage must be 1-100", 422)
        strategy = self.resources.create(
            "agent_release_strategies",
            tenant_id,
            user_id,
            {
                "name": f"gray {agent.name}",
                "status": "active",
                "parent_id": agent.id,
                "agent_id": agent.id,
                "version": str(payload.get("version") or agent.version),
                "config": {"percentage": percentage, "rules": payload.get("rules", [])},
                "spec": payload,
            },
        )
        return ResourceService.to_dict(strategy)

    def rollback(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        target_version = str(payload.get("target_version") or "")
        if not target_version:
            releases, _ = self.resources.list("agent_releases", tenant_id, 1, 20, {"agent_id": agent.id})
            target_version = next((row.version for row in releases if row.version and row.version != agent.version), "")
        if not target_version:
            raise AppError(ErrorCode.NOT_FOUND, "rollback target version not found", 404)
        release = self.resources.create(
            "agent_releases",
            tenant_id,
            user_id,
            {
                "name": f"rollback {agent.name} to {target_version}",
                "status": "rolled_back",
                "parent_id": agent.id,
                "agent_id": agent.id,
                "version": target_version,
                "spec": {"from_version": agent.version, "target_version": target_version, **payload},
            },
        )
        self.resources.update("agents", tenant_id, user_id, agent.id, {"status": "released", "version": target_version})
        return ResourceService.to_dict(release)

    def import_agent(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        spec = dict(payload.get("agent") or payload)
        agent = self.resources.create(
            "agents",
            tenant_id,
            user_id,
            {
                "name": str(spec.get("name") or "Imported Agent"),
                "code": str(spec.get("code") or f"imported-{datetime.now(timezone.utc).strftime('%H%M%S')}"),
                "description": str(spec.get("description") or ""),
                "status": "draft",
                "model_type": str(spec.get("model_type") or ""),
                "config": dict(spec.get("config") or {}),
                "spec": dict(spec.get("spec") or {}),
            },
        )
        record = self.resources.create(
            "agent_import_exports",
            tenant_id,
            user_id,
            {"name": "agent import", "status": "completed", "agent_id": agent.id, "output_payload": {"agent_id": agent.id}},
        )
        return {"agent": ResourceService.to_dict(agent), "record_id": record.id}

    def export_agent(self, tenant_id: str, user_id: str, agent_id: str) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        versions, _ = self.resources.list("agent_versions", tenant_id, 1, 200, {"agent_id": agent.id})
        releases, _ = self.resources.list("agent_releases", tenant_id, 1, 200, {"agent_id": agent.id})
        payload = {
            "agent": ResourceService.to_dict(agent),
            "versions": [ResourceService.to_dict(row) for row in versions],
            "releases": [ResourceService.to_dict(row) for row in releases],
        }
        record = self.resources.create(
            "agent_import_exports",
            tenant_id,
            user_id,
            {"name": f"export {agent.name}", "status": "completed", "agent_id": agent.id, "output_payload": payload},
        )
        return {"record_id": record.id, **payload}

    async def debug_run(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = await AgentRunnerService(self.db).run(tenant_id, user_id, agent_id, payload)
        session = self.resources.create(
            "agent_debug_sessions",
            tenant_id,
            user_id,
            {"name": "agent debug", "status": "completed", "agent_id": agent_id, "session_id": str(payload.get("session_id") or "debug"), "input_payload": payload, "output_payload": result},
        )
        return {"debug_session_id": session.id, **result}

    def status(self, tenant_id: str, agent_id: str) -> dict[str, Any]:
        agent = self.resources.get("agents", tenant_id, agent_id)
        instances, total = self.resources.list("agent_runtime_instances", tenant_id, 1, 200, {"agent_id": agent.id})
        runs, run_total = self.resources.list("agent_run_records", tenant_id, 1, 1, {"agent_id": agent.id})
        return {
            "agent_id": agent.id,
            "status": agent.status,
            "enabled": agent.enabled,
            "version": agent.version,
            "runtime_instances": total,
            "latest_run": ResourceService.to_dict(runs[0]) if runs else None,
            "run_total": run_total,
        }

    def _latest_version(self, tenant_id: str, user_id: str, agent_id: str) -> Any:
        versions, _ = self.resources.list("agent_versions", tenant_id, 1, 1, {"agent_id": agent_id})
        if versions:
            return versions[0]
        agent = self.resources.get("agents", tenant_id, agent_id)
        return self.resources.create(
            "agent_versions",
            tenant_id,
            user_id,
            {
                "name": f"{agent.name} v1",
                "status": "draft",
                "parent_id": agent.id,
                "agent_id": agent.id,
                "version": "v1",
                "model_type": agent.model_type,
                "config": agent.config,
                "spec": {"agent": ResourceService.to_dict(agent)},
            },
        )
