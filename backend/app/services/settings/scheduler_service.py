from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.services._shared.resource_service import ResourceService


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

        self._refresh_next_run(tenant_id, user_id, job.id)

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
        trigger_type = str((payload or {}).get("_trigger_type") or "manual")

        execution = self.resources.create("scheduler_executions", tenant_id, user_id, {
            "name": f"exec-{uuid.uuid4().hex[:6]}",
            "code": f"se-{uuid.uuid4().hex[:8]}",
            "status": "running",
            "parent_id": job_id,
            "agent_id": target_id if target_type == "agent" else "",
            "workflow_id": target_id if target_type == "workflow" else "",
            "spec": {
                "trigger_type": trigger_type,
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
            self._refresh_next_run(tenant_id, user_id, job_id)

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
            self._refresh_next_run(tenant_id, user_id, job_id)

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
        if not target_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "scheduler target_id is required", 422)
        if target_type == "workflow":
            from app.services.build.workflow_service import WorkflowService

            return await WorkflowService(self.db).run(tenant_id, user_id, target_id, payload)
        if target_type == "agent":
            from app.services.runtime.agent_runner_service import AgentRunnerService

            return await AgentRunnerService(self.db).run(tenant_id, user_id, target_id, payload)
        raise AppError(ErrorCode.VALIDATION_ERROR, f"unsupported scheduler target_type: {target_type}", 422)

    def _schedule_next_run(self, tenant_id: str, user_id: str, job_id: str, cron_expression: str) -> None:
        job = self.resources.get("scheduler_jobs", tenant_id, job_id)
        spec = dict(job.spec or {})
        spec["cron_expression"] = cron_expression
        spec["next_run_at"] = self._next_run_at(spec)
        self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, {"spec": spec})

    def _refresh_next_run(self, tenant_id: str, user_id: str, job_id: str) -> None:
        job = self.resources.get("scheduler_jobs", tenant_id, job_id)
        spec = dict(job.spec or {})
        spec["next_run_at"] = self._next_run_at(spec)
        self.resources.update("scheduler_jobs", tenant_id, user_id, job_id, {"spec": spec})

    async def run_due_jobs(self, tenant_id: str, user_id: str = "system", limit: int = 100) -> dict[str, Any]:
        jobs, _ = self.resources.list("scheduler_jobs", tenant_id, 1, limit, {"status": "active"})
        now = datetime.now(timezone.utc)
        executed: list[dict[str, Any]] = []
        for job in jobs:
            spec = dict(job.spec or {})
            if not spec.get("enabled", True):
                continue
            next_run_at = self._parse_time(str(spec.get("next_run_at") or ""))
            if next_run_at and next_run_at <= now:
                result = await self.trigger_job(tenant_id, user_id, job.id, {"_trigger_type": "schedule", **dict(spec.get("input_payload") or {})})
                executed.append({"job_id": job.id, **result})
        return {"status": "completed", "executed": executed, "count": len(executed)}

    def _next_run_at(self, spec: dict[str, Any]) -> str | None:
        if not spec.get("enabled", True):
            return None
        job_type = str(spec.get("job_type") or "cron")
        now = datetime.now(timezone.utc)
        if job_type == "interval":
            interval = int(spec.get("interval_seconds") or 0)
            if interval <= 0:
                raise AppError(ErrorCode.VALIDATION_ERROR, "interval job requires interval_seconds", 422)
            return (now + timedelta(seconds=interval)).isoformat()
        if job_type == "cron":
            expression = str(spec.get("cron_expression") or "")
            if not expression:
                raise AppError(ErrorCode.VALIDATION_ERROR, "cron job requires cron_expression", 422)
            return self._next_cron_time(expression, now).isoformat()
        return None

    def _next_cron_time(self, expression: str, after: datetime) -> datetime:
        fields = expression.split()
        if len(fields) != 5:
            raise AppError(ErrorCode.VALIDATION_ERROR, "cron_expression must have five fields", 422)
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(366 * 24 * 60):
            if (
                self._cron_matches(fields[0], candidate.minute)
                and self._cron_matches(fields[1], candidate.hour)
                and self._cron_matches(fields[2], candidate.day)
                and self._cron_matches(fields[3], candidate.month)
                and self._cron_matches(fields[4], candidate.isoweekday() % 7)
            ):
                return candidate
            candidate += timedelta(minutes=1)
        raise AppError(ErrorCode.VALIDATION_ERROR, "cron_expression did not match within one year", 422)

    @staticmethod
    def _cron_matches(field: str, value: int) -> bool:
        for part in field.split(","):
            part = part.strip()
            if part == "*":
                return True
            if part.startswith("*/"):
                step = int(part[2:])
                if step > 0 and value % step == 0:
                    return True
            elif "-" in part:
                start, end = [int(item) for item in part.split("-", 1)]
                if start <= value <= end:
                    return True
            elif part and int(part) == value:
                return True
        return False

    @staticmethod
    def _parse_time(value: str) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

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
