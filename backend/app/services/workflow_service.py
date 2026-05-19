from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.services.knowledge_service import KnowledgeService
from app.services.model_services import ModelInvocationService
from app.services.resource_service import ResourceService
from app.services.tool_service import ToolService

project_root = settings.project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from runtime.executors.workflow_executor import WorkflowExecutor, WorkflowValidationError  # noqa: E402


class WorkflowService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)
        self.executor = WorkflowExecutor()

    def create_version(self, tenant_id: str, user_id: str, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        workflow = self.resources.get("workflow_definitions", tenant_id, workflow_id)
        version = str(payload.get("version") or workflow.version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        row = self.resources.create(
            "workflow_versions",
            tenant_id,
            user_id,
            {
                "name": f"{workflow.name} {version}",
                "status": "draft",
                "parent_id": workflow.id,
                "workflow_id": workflow.id,
                "version": version,
                "config": workflow.config,
                "spec": {"workflow": ResourceService.to_dict(workflow), "nodes": self._nodes(workflow, payload), "edges": self._edges(workflow, payload)},
            },
        )
        return ResourceService.to_dict(row)

    def validate(self, tenant_id: str, user_id: str, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        workflow = self.resources.get("workflow_definitions", tenant_id, workflow_id)
        nodes = self._nodes(workflow, payload)
        edges = self._edges(workflow, payload)
        try:
            result = self.executor.validate(nodes, edges)
            status = "valid"
            error_message = ""
        except WorkflowValidationError as exc:
            result = {"valid": False, "error": str(exc)}
            status = "invalid"
            error_message = str(exc)
        self.resources.create(
            "workflow_run_events",
            tenant_id,
            user_id,
            {
                "name": "workflow validation",
                "status": status,
                "workflow_id": workflow.id,
                "input_payload": {"nodes": nodes, "edges": edges},
                "output_payload": result,
                "error_message": error_message,
            },
        )
        return result

    def publish(self, tenant_id: str, user_id: str, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate(tenant_id, user_id, workflow_id, payload)
        if not validation.get("valid"):
            raise AppError(ErrorCode.VALIDATION_ERROR, str(validation.get("error", "workflow invalid")), 422)
        version = self.create_version(tenant_id, user_id, workflow_id, {**payload, "version": payload.get("version") or "published"})
        self.resources.update("workflow_definitions", tenant_id, user_id, workflow_id, {"status": "published", "version": str(version["version"])})
        version["status"] = "published"
        return version

    async def run(self, tenant_id: str, user_id: str, workflow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        workflow = self.resources.get("workflow_definitions", tenant_id, workflow_id)
        nodes = self._nodes(workflow, payload)
        edges = self._edges(workflow, payload)
        plan = self.executor.validate(nodes, edges)
        run = self.resources.create(
            "workflow_runs",
            tenant_id,
            user_id,
            {
                "name": f"run {workflow.name}",
                "status": "running",
                "workflow_id": workflow.id,
                "input_payload": payload,
                "spec": {"nodes": nodes, "edges": edges, "order": plan["order"]},
                "started_at": datetime.now(timezone.utc),
            },
        )
        context = dict(payload.get("context") or {})
        results: dict[str, Any] = {}
        node_map = {str(node["id"]): node for node in nodes}
        try:
            for node_id in plan["order"]:
                node = node_map[node_id]
                step = self.resources.create(
                    "workflow_run_steps",
                    tenant_id,
                    user_id,
                    {"name": str(node.get("label") or node_id), "status": "running", "workflow_id": workflow.id, "run_id": run.id, "spec": {"node": node}},
                )
                result = await self._execute_node(tenant_id, user_id, workflow.id, run.id, node, context)
                if result.get("status") == "waiting_approval":
                    self.resources.update("workflow_run_steps", tenant_id, user_id, step.id, {"status": "waiting_approval", "output_payload": result})
                    self.resources.update("workflow_runs", tenant_id, user_id, run.id, {"status": "waiting_approval", "output_payload": {"results": results, "waiting": result}})
                    return {"run_id": run.id, "status": "waiting_approval", "waiting": result, "results": results}
                results[node_id] = result
                context[node_id] = result
                self.resources.update("workflow_run_steps", tenant_id, user_id, step.id, {"status": "success", "output_payload": result})
                self.resources.create(
                    "workflow_run_logs",
                    tenant_id,
                    user_id,
                    {"name": f"node {node_id}", "status": "success", "workflow_id": workflow.id, "run_id": run.id, "parent_id": step.id, "output_payload": result},
                )
            self.resources.update(
                "workflow_runs",
                tenant_id,
                user_id,
                run.id,
                {"status": "success", "finished_at": datetime.now(timezone.utc), "output_payload": {"results": results, "context": context}},
            )
            return {"run_id": run.id, "status": "success", "results": results, "context": context}
        except Exception as exc:
            self.resources.update(
                "workflow_runs",
                tenant_id,
                user_id,
                run.id,
                {"status": "failed", "finished_at": datetime.now(timezone.utc), "error_message": str(exc), "output_payload": {"results": results}},
            )
            self.resources.create(
                "workflow_run_events",
                tenant_id,
                user_id,
                {"name": "workflow failed", "status": "failed", "workflow_id": workflow.id, "run_id": run.id, "error_message": str(exc)},
            )
            raise

    async def retry(self, tenant_id: str, user_id: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run = self.resources.get("workflow_runs", tenant_id, run_id)
        if not run.workflow_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "run has no workflow_id", 422)
        retry_payload = {**(run.input_payload or {}), **payload, "context": payload.get("context") or (run.input_payload or {}).get("context") or {}}
        return await self.run(tenant_id, user_id, run.workflow_id, retry_payload)

    def cancel(self, tenant_id: str, user_id: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run = self.resources.update("workflow_runs", tenant_id, user_id, run_id, {"status": "cancelled", "finished_at": datetime.now(timezone.utc), "output_payload": payload})
        event = self.resources.create("workflow_run_events", tenant_id, user_id, {"name": "workflow cancelled", "status": "cancelled", "run_id": run.id, "workflow_id": run.workflow_id})
        return {"run_id": run.id, "event_id": event.id, "status": "cancelled"}

    def replay(self, tenant_id: str, user_id: str, run_id: str) -> dict[str, Any]:
        run = self.resources.get("workflow_runs", tenant_id, run_id)
        event = self.resources.create(
            "workflow_run_events",
            tenant_id,
            user_id,
            {"name": "workflow replay", "status": "completed", "run_id": run.id, "workflow_id": run.workflow_id, "output_payload": run.output_payload},
        )
        return {"run_id": run.id, "event_id": event.id, "status": "completed", "replay": run.output_payload}

    def decide_approval(self, tenant_id: str, user_id: str, approval_id: str, approved: bool, payload: dict[str, Any]) -> dict[str, Any]:
        approval = self.resources.update(
            "workflow_approvals",
            tenant_id,
            user_id,
            approval_id,
            {"status": "approved" if approved else "rejected", "output_payload": payload, "finished_at": datetime.now(timezone.utc)},
        )
        event = self.resources.create(
            "workflow_run_events",
            tenant_id,
            user_id,
            {
                "name": "approval decided",
                "status": approval.status,
                "workflow_id": approval.workflow_id,
                "run_id": approval.run_id,
                "parent_id": approval.id,
                "output_payload": {"approved": approved, "payload": payload},
            },
        )
        return {"approval_id": approval.id, "event_id": event.id, "status": approval.status, "approved": approved}

    async def _execute_node(self, tenant_id: str, user_id: str, workflow_id: str, run_id: str, node: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        node_type = str(node.get("type") or node.get("shape") or "script")
        config = dict(node.get("config") or node.get("data") or {})
        if node_type in {"start", "end", "aggregate", "parallel"}:
            return {"status": "success", "type": node_type, "context_keys": sorted(context)}
        if node_type in {"approval", "human_approval"}:
            approval = self.resources.create(
                "workflow_approvals",
                tenant_id,
                user_id,
                {
                    "name": str(node.get("label") or "approval"),
                    "status": "pending",
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "spec": {"node": node, "approvers": config.get("approvers", [])},
                },
            )
            return {"status": "waiting_approval", "approval_id": approval.id, "type": node_type}
        if node_type in {"condition", "branch"}:
            left = context.get(str(config.get("left") or "input"), config.get("left_value"))
            right = config.get("right_value")
            operator = str(config.get("operator") or "equals")
            matched = self._compare(left, right, operator)
            return {"status": "success", "type": node_type, "matched": matched, "operator": operator}
        if node_type in {"model", "chat_llm", "vision_language", "embedding", "rerank"} and config.get("model_id"):
            model_type = str(config.get("model_type") or node_type)
            invocation = await ModelInvocationService(self.db).invoke(tenant_id, user_id, str(config["model_id"]), model_type, dict(config.get("payload") or context))
            return {"status": "success", "type": node_type, "invocation": invocation}
        if node_type in {"rag", "rag_retrieve"} and config.get("knowledge_base_id"):
            retrieval = KnowledgeService(self.db).retrieve(tenant_id, user_id, str(config["knowledge_base_id"]), dict(config.get("payload") or context))
            return {"status": "success", "type": node_type, "retrieval": retrieval}
        if node_type == "transform":
            return {"status": "success", "type": node_type, "output": self._transform(config, context)}
        if node_type == "script":
            if config.get("output") is not None:
                return {"status": "success", "type": node_type, "output": config.get("output")}
            raise AppError(ErrorCode.VALIDATION_ERROR, "script node requires a sandboxed tool_id; inline script execution is disabled", 422)
        if node_type == "http":
            return await self._execute_http_node(config, context)
        if node_type == "tool":
            tool_id = str(config.get("tool_id") or config.get("id") or "")
            if not tool_id:
                raise AppError(ErrorCode.VALIDATION_ERROR, "tool node requires tool_id", 422)
            result = await ToolService(self.db).invoke(tenant_id, user_id, tool_id, {"arguments": dict(config.get("arguments") or config.get("payload") or context)})
            return {"status": "success", "type": node_type, "tool": result}
        if node_type == "agent":
            agent_id = str(config.get("agent_id") or "")
            if not agent_id:
                raise AppError(ErrorCode.VALIDATION_ERROR, "agent node requires agent_id", 422)
            from app.runtime.service import RuntimeControlService

            prompt = str(config.get("prompt") or context.get("input") or context.get("prompt") or context)
            result = await RuntimeControlService(self.db).debug_run(tenant_id, user_id, agent_id, {"prompt": prompt, "session_id": run_id, **dict(config.get("runtime") or {})})
            return {"status": "success", "type": node_type, "agent": result}
        raise AppError(ErrorCode.VALIDATION_ERROR, f"unsupported workflow node type: {node_type}", 422)

    async def _execute_http_node(self, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        url = str(config.get("url") or "")
        if not url.startswith(("http://", "https://")):
            raise AppError(ErrorCode.VALIDATION_ERROR, "http node requires url", 422)
        method = str(config.get("method") or "POST").upper()
        headers = dict(config.get("headers") or {})
        body = config.get("body", config.get("payload", context))
        async with httpx.AsyncClient(timeout=float(config.get("timeout_seconds") or 30)) as client:
            response = await client.request(method, url, headers=headers, json=body if method not in {"GET", "HEAD"} else None)
        payload: Any
        if "application/json" in response.headers.get("content-type", ""):
            payload = response.json()
        else:
            payload = response.text
        if response.status_code >= 400:
            raise AppError(ErrorCode.BUSINESS_ERROR, f"http node failed with status {response.status_code}", response.status_code)
        return {"status": "success", "type": "http", "status_code": response.status_code, "output": payload}

    @staticmethod
    def _transform(config: dict[str, Any], context: dict[str, Any]) -> Any:
        if "output" in config:
            return config["output"]
        pick = config.get("pick")
        if isinstance(pick, list):
            return {str(key): context.get(str(key)) for key in pick}
        if isinstance(pick, str) and pick:
            return context.get(pick)
        merge = dict(config.get("merge") or {})
        return {**context, **merge}

    @staticmethod
    def _nodes(workflow: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = payload.get("nodes") or (workflow.spec or {}).get("nodes") or (workflow.config or {}).get("nodes")
        if not isinstance(nodes, list):
            raise AppError(ErrorCode.VALIDATION_ERROR, "workflow nodes are required", 422)
        return [dict(node) for node in nodes]

    @staticmethod
    def _edges(workflow: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
        edges = payload.get("edges") or (workflow.spec or {}).get("edges") or (workflow.config or {}).get("edges") or []
        if not isinstance(edges, list):
            raise AppError(ErrorCode.VALIDATION_ERROR, "workflow edges must be a list", 422)
        return [dict(edge) for edge in edges]

    @staticmethod
    def _compare(left: Any, right: Any, operator: str) -> bool:
        if operator == "contains":
            return str(right) in str(left)
        if operator == "not_equals":
            return left != right
        if operator == "gt":
            return float(left) > float(right)
        if operator == "lt":
            return float(left) < float(right)
        return left == right
