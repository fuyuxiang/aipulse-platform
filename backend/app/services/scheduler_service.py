from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class SchedulerService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def create_job(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name", ""))
        job_type = str(payload.get("job_type", "cron"))
        target_type = str(payload.get("target_type", "agent"))
        target_id = str(payload.get("target_id", ""))

        job = self.resources.create("scheduler_jobs", tenant_id, user_id, {
            "name": name,
            "code": f"job-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": target_id if target_type == "agent" else "",
            "workflow_id": target_id if target_type == "workflow" else "",
            "spec": {
                "job_type": job_type,
                "target_type": target_type,
                "target_id": target_id,
                "description": str(payload.get("description", "")),
                "cron_expression": str(payload.get("cron_expression", "")) if job_type == "cron" else "",
                "interval_seconds": int(payload.get("interval_seconds", 0)) if job_type == "interval" else 0,
                "event_type": str(payload.get("event_type", "")) if job_type == "event" else "",
                "event_filter": payload.get("event_filter", {}) if job_type == "event" else {},
                "input_payload": payload.get("input_payload", {}),
                "retry_policy": payload.get("retry_policy", {
                    "max_retries": 3,
                    "retry_delay_seconds": 60,
                    "backoff_multiplier": 2.0,
                }),
                "timeout_seconds": int(payload.get("timeout_seconds", 300)),
                "max_concurrent": int(payload.get("max_concurrent", 1)),
                "enabled": True,
                "next_run_at": None,
                "last_run_at": None,
                "total_runs": 0,
                "success_count": 0,
                "failure_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        if job_type == "cron" and payload.get("cron_expression"):
            self._schedule_next_run(tenant_id, user_id, job.id, str(payload["cron_expression"]))

        return ResourceService.to_dict(job)

    def update_job(self, tenant_id: str, user_id: str, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, payload)
        return ResourceService.to_dict(row)

    def delete_job(self, tenant_id: str, user_id: str, job_id: str) -> dict[str, str]:
        return self.resources.delete("scheduler_jobs", tenant_id, user_id, job_id)

    def get_job(self, tenant_id: str, job_id: str) -> dict[str, Any]:
        job = ResourceService.to_dict(self.resources.get("scheduler_jobs", tenant_id, job_id))
        executions, total = self.resources.list("scheduler_executions", tenant_id, 1, 10, {"parent_id": job_id})
        job["recent_executions"] = [ResourceService.to_dict(e) for e in executions]
        job["execution_count"] = total
        return job

    def list_jobs(self, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("scheduler_jobs", tenant_id, page, page_size, filters)
        return [ResourceService.to_dict(row) for row in rows], total

    def enable_job(self, tenant_id: str, user_id: str, job_id: str) -> dict[str, Any]:
        job = self.resources.get("scheduler_jobs", tenant_id, job_id)
        spec = dict(job.spec or {})
        spec["enabled"] = True
        row = self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, {"status": "active", "spec": spec})
        return ResourceService.to_dict(row)

    def disable_job(self, tenant_id: str, user_id: str, job_id: str) -> dict[str, Any]:
        job = self.resources.get("scheduler_jobs", tenant_id, job_id)
        spec = dict(job.spec or {})
        spec["enabled"] = False
        row = self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, {"status": "paused", "spec": spec})
        return ResourceService.to_dict(row)

    async def trigger_job(self, tenant_id: str, user_id: str, job_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        job = self.resources.get("scheduler_jobs", tenant_id, job_id)
        spec = dict(job.spec or {})
        target_type = spec.get("target_type", "agent")
        target_id = spec.get("target_id", "")
        input_payload = payload or spec.get("input_payload", {})

        execution = self.resources.create("scheduler_executions", tenant_id, user_id, {
            "name": f"exec-{uuid.uuid4().hex[:6]}",
            "code": f"se-{uuid.uuid4().hex[:8]}",
            "status": "running",
            "parent_id": job_id,
            "agent_id": target_id if target_type == "agent" else "",
            "workflow_id": target_id if target_type == "workflow" else "",
            "spec": {
                "trigger_type": "manual",
                "input_payload": input_payload,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        try:
            result = await self._execute_target(tenant_id, user_id, target_type, target_id, input_payload)
            exec_spec = dict(execution.spec or {})
            exec_spec["finished_at"] = datetime.now(timezone.utc).isoformat()
            exec_spec["result"] = result
            self.resources.update("scheduler_executions", tenant_id, user_id, execution.id, {"status": "completed", "spec": exec_spec})

            spec["last_run_at"] = datetime.now(timezone.utc).isoformat()
            spec["total_runs"] = spec.get("total_runs", 0) + 1
            spec["success_count"] = spec.get("success_count", 0) + 1
            self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, {"spec": spec})

            return {"execution_id": execution.id, "status": "completed", "result": result}
        except Exception as e:
            exec_spec = dict(execution.spec or {})
            exec_spec["finished_at"] = datetime.now(timezone.utc).isoformat()
            exec_spec["error"] = str(e)
            self.resources.update("scheduler_executions", tenant_id, user_id, execution.id, {"status": "failed", "spec": exec_spec})

            spec["last_run_at"] = datetime.now(timezone.utc).isoformat()
            spec["total_runs"] = spec.get("total_runs", 0) + 1
            spec["failure_count"] = spec.get("failure_count", 0) + 1
            self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, {"spec": spec})

            return {"execution_id": execution.id, "status": "failed", "error": str(e)}

    def list_executions(self, tenant_id: str, job_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("scheduler_executions", tenant_id, page, page_size, {"parent_id": job_id})
        return [ResourceService.to_dict(row) for row in rows], total

    def create_webhook(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        secret = uuid.uuid4().hex
        webhook = self.resources.create("scheduler_webhooks", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"wh-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": str(payload.get("target_id", "")) if payload.get("target_type") == "agent" else "",
            "workflow_id": str(payload.get("target_id", "")) if payload.get("target_type") == "workflow" else "",
            "spec": {
                "target_type": str(payload.get("target_type", "agent")),
                "target_id": str(payload.get("target_id", "")),
                "secret": secret,
                "url_path": f"/api/v1/webhooks/{uuid.uuid4().hex[:12]}",
                "method": str(payload.get("method", "POST")),
                "headers_filter": payload.get("headers_filter", {}),
                "body_template": payload.get("body_template", {}),
                "enabled": True,
                "total_calls": 0,
                "last_called_at": None,
                "description": str(payload.get("description", "")),
            },
        })
        return ResourceService.to_dict(webhook)

    def update_webhook(self, tenant_id: str, user_id: str, webhook_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("scheduler_webhooks", tenant_id, user_id, webhook_id, payload)
        return ResourceService.to_dict(row)

    def delete_webhook(self, tenant_id: str, user_id: str, webhook_id: str) -> dict[str, str]:
        return self.resources.delete("scheduler_webhooks", tenant_id, user_id, webhook_id)

    def get_webhook(self, tenant_id: str, webhook_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("scheduler_webhooks", tenant_id, webhook_id))

    def list_webhooks(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("scheduler_webhooks", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    async def handle_webhook(self, tenant_id: str, webhook_id: str, payload: dict[str, Any], signature: str = "") -> dict[str, Any]:
        webhook = self.resources.get("scheduler_webhooks", tenant_id, webhook_id)
        spec = dict(webhook.spec or {})

        if not spec.get("enabled", True):
            return {"status": "disabled", "message": "Webhook is disabled"}

        secret = spec.get("secret", "")
        if secret and signature:
            import json
            expected = hmac.new(secret.encode(), json.dumps(payload).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return {"status": "unauthorized", "message": "Invalid signature"}

        target_type = spec.get("target_type", "agent")
        target_id = spec.get("target_id", "")

        result = await self._execute_target(tenant_id, "system", target_type, target_id, payload)

        spec["total_calls"] = spec.get("total_calls", 0) + 1
        spec["last_called_at"] = datetime.now(timezone.utc).isoformat()
        self.resources.update("scheduler_webhooks", tenant_id, "system", webhook.id, {"spec": spec})

        return {"status": "executed", "result": result}

    def create_trigger(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        trigger = self.resources.create("scheduler_triggers", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"trg-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "spec": {
                "event_source": str(payload.get("event_source", "")),
                "event_type": str(payload.get("event_type", "")),
                "conditions": payload.get("conditions", []),
                "target_type": str(payload.get("target_type", "agent")),
                "target_id": str(payload.get("target_id", "")),
                "transform_template": payload.get("transform_template", {}),
                "enabled": True,
                "description": str(payload.get("description", "")),
            },
        })
        return ResourceService.to_dict(trigger)

    def list_triggers(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("scheduler_triggers", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def update_trigger(self, tenant_id: str, user_id: str, trigger_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("scheduler_triggers", tenant_id, user_id, trigger_id, payload)
        return ResourceService.to_dict(row)

    def delete_trigger(self, tenant_id: str, user_id: str, trigger_id: str) -> dict[str, str]:
        return self.resources.delete("scheduler_triggers", tenant_id, user_id, trigger_id)

    async def _execute_target(self, tenant_id: str, user_id: str, target_type: str, target_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if target_type == "workflow":
                from app.services.workflow_service import WorkflowService
                return await WorkflowService(self.db).run(tenant_id, user_id, target_id, payload)
            else:
                from app.runtime.service import RuntimeControlService
                return await RuntimeControlService(self.db).debug_run(tenant_id, user_id, target_id, payload)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _schedule_next_run(self, tenant_id: str, user_id: str, job_id: str, cron_expression: str) -> None:
        pass

    def get_stats(self, tenant_id: str) -> dict[str, Any]:
        _, total_jobs = self.resources.list("scheduler_jobs", tenant_id, 1, 1)
        _, total_executions = self.resources.list("scheduler_executions", tenant_id, 1, 1)
        _, total_webhooks = self.resources.list("scheduler_webhooks", tenant_id, 1, 1)
        _, total_triggers = self.resources.list("scheduler_triggers", tenant_id, 1, 1)
        return {
            "total_jobs": total_jobs,
            "total_executions": total_executions,
            "total_webhooks": total_webhooks,
            "total_triggers": total_triggers,
        }
