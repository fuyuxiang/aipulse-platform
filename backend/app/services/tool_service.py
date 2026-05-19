from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.services.resource_service import ResourceService


class ToolService:
    def __init__(self, db: Session):
        self.resources = ResourceService(db)

    async def invoke(self, tenant_id: str, user_id: str, tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self.resources.get("tools", tenant_id, tool_id)
        arguments = dict(payload.get("arguments") or payload.get("payload") or {})
        self._validate_schema(dict((tool.spec or {}).get("schema") or (tool.config or {}).get("schema") or {}), arguments)
        self._check_rate_limit(tenant_id, tool)
        if self._requires_approval(tool, payload):
            approval = self.resources.create(
                "tool_approval_tasks",
                tenant_id,
                user_id,
                {
                    "name": f"approve {tool.name}",
                    "status": "pending",
                    "parent_id": tool.id,
                    "tool_name": tool.name,
                    "input_payload": {"arguments": arguments, "reason": payload.get("reason", "")},
                    "spec": {"risk_level": self._risk_level(tool), "tool_id": tool.id},
                },
            )
            return {"status": "waiting_approval", "approval_id": approval.id, "tool_id": tool.id}
        started = time.perf_counter()
        try:
            output = await self._execute(tool, arguments)
            status = "success"
            error_message = ""
        except Exception as exc:
            output = {"error": str(exc)}
            status = "failed"
            error_message = str(exc)
        latency_ms = int((time.perf_counter() - started) * 1000)
        log = self.resources.create(
            "tool_call_logs",
            tenant_id,
            user_id,
            {
                "name": f"invoke {tool.name}",
                "status": status,
                "parent_id": tool.id,
                "tool_name": tool.name,
                "latency_ms": latency_ms,
                "input_payload": {"arguments": arguments},
                "output_payload": output,
                "error_message": error_message,
            },
        )
        if status == "failed":
            raise AppError(ErrorCode.BUSINESS_ERROR, error_message, 500)
        return {"status": status, "tool_id": tool.id, "log_id": log.id, "latency_ms": latency_ms, "output": output}

    def approve_task(self, tenant_id: str, user_id: str, approval_id: str, approved: bool, payload: dict[str, Any]) -> dict[str, Any]:
        approval = self.resources.update(
            "tool_approval_tasks",
            tenant_id,
            user_id,
            approval_id,
            {"status": "approved" if approved else "rejected", "output_payload": payload, "finished_at": datetime.now(timezone.utc)},
        )
        log = self.resources.create(
            "tool_call_logs",
            tenant_id,
            user_id,
            {
                "name": "tool approval",
                "status": approval.status,
                "parent_id": approval.parent_id,
                "tool_name": approval.tool_name,
                "input_payload": approval.input_payload,
                "output_payload": {"approved": approved, "decision": payload},
            },
        )
        return {"approval_id": approval.id, "status": approval.status, "approved": approved, "log_id": log.id}

    def sync_mcp_tools(self, tenant_id: str, user_id: str, server_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        server = self.resources.get("mcp_servers", tenant_id, server_id)
        tool_specs = payload.get("tools") or (server.spec or {}).get("tools") or []
        if not isinstance(tool_specs, list):
            raise AppError(ErrorCode.VALIDATION_ERROR, "tools must be a list", 422)
        created = []
        for item in tool_specs:
            spec = dict(item)
            row = self.resources.create(
                "mcp_tools",
                tenant_id,
                user_id,
                {
                    "name": str(spec.get("name") or spec.get("code") or "mcp-tool"),
                    "code": str(spec.get("code") or spec.get("name") or ""),
                    "status": "active",
                    "parent_id": server.id,
                    "tool_name": str(spec.get("name") or ""),
                    "spec": spec,
                },
            )
            created.append(row.id)
        return {"server_id": server.id, "created": created, "count": len(created), "status": "completed"}

    async def _execute(self, tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        config = dict(tool.config or {})
        kind = str(config.get("type") or (tool.spec or {}).get("type") or "echo")
        timeout = float(config.get("timeout_seconds") or 10)
        if kind == "echo":
            return {"arguments": arguments}
        if kind == "calculator":
            numbers = [float(item) for item in arguments.get("numbers", [])]
            operation = str(arguments.get("operation") or config.get("operation") or "sum")
            value = 1.0
            if operation == "multiply":
                for number in numbers:
                    value *= number
            else:
                value = sum(numbers)
            return {"operation": operation, "value": value}
        if kind == "template":
            template = str(config.get("template") or "{value}")
            return {"text": template.format(**arguments)}
        if kind == "http":
            method = str(config.get("method") or arguments.get("method") or "GET").upper()
            url = str(config.get("url") or arguments.get("url") or "")
            if not url.startswith(("http://", "https://")):
                raise AppError(ErrorCode.VALIDATION_ERROR, "http tool requires url", 422)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, json=arguments.get("json"))
            return {"status_code": response.status_code, "text": response.text[:2000], "headers": dict(response.headers)}
        if kind == "sleep":
            await asyncio.sleep(min(timeout, float(arguments.get("seconds", 0))))
            return {"slept": arguments.get("seconds", 0)}
        raise AppError(ErrorCode.VALIDATION_ERROR, f"unsupported tool type: {kind}", 422)

    def _check_rate_limit(self, tenant_id: str, tool: Any) -> None:
        limit = int((tool.config or {}).get("rate_limit_per_minute") or 0)
        if limit <= 0:
            return
        rows, _ = self.resources.list("tool_call_logs", tenant_id, 1, 1000, {"tool_name": tool.name})
        recent = [row for row in rows if (datetime.now(timezone.utc) - row.created_at).total_seconds() < 60]
        if len(recent) >= limit:
            raise AppError(ErrorCode.RATE_LIMITED, "tool rate limit exceeded", 429)

    def _requires_approval(self, tool: Any, payload: dict[str, Any]) -> bool:
        if payload.get("approval_id") or payload.get("approved"):
            return False
        risk = self._risk_level(tool)
        return bool((tool.spec or {}).get("requires_approval") or risk in {"high", "critical"})

    @staticmethod
    def _risk_level(tool: Any) -> str:
        return str((tool.spec or {}).get("risk_level") or (tool.config or {}).get("risk_level") or "low").lower()

    @staticmethod
    def _validate_schema(schema: dict[str, Any], arguments: dict[str, Any]) -> None:
        if not schema:
            return
        required = schema.get("required") or []
        for key in required:
            if key not in arguments:
                raise AppError(ErrorCode.VALIDATION_ERROR, f"missing required tool argument: {key}", 422)
        properties = dict(schema.get("properties") or {})
        type_map: dict[str, Any] = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}
        for key, rule in properties.items():
            if key not in arguments:
                continue
            expected = type_map.get(str((rule or {}).get("type") or ""))
            if expected and not isinstance(arguments[key], expected):
                raise AppError(ErrorCode.VALIDATION_ERROR, f"tool argument {key} must be {rule['type']}", 422)
