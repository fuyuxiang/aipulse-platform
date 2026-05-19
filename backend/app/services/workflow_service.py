from __future__ import annotations

import asyncio
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
from runtime.executors.workflow_checkpoint import CheckpointStore, build_checkpoint  # noqa: E402
from runtime.executors.workflow_compensation import CompensationEngine, CompensationStrategy  # noqa: E402
from runtime.executors.workflow_events import EventBus, get_event_bus  # noqa: E402


class WorkflowService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)
        self.executor = WorkflowExecutor()
        self._checkpoint_store = CheckpointStore(settings.resolved_data_dir / "checkpoints")
        self._event_bus = get_event_bus()

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
        handlers = self._build_handlers(tenant_id, user_id, workflow.id, run.id)
        compensation_strategy = str(payload.get("compensation_strategy", "backward_recovery"))
        timeout_seconds = float(payload.get("timeout_seconds", 0))

        async def checkpoint_cb(cp: dict[str, Any]) -> None:
            checkpoint = build_checkpoint(
                run_id=run.id,
                results=cp.get("results", {}),
                context=cp.get("context", {}),
                compensation_stack=cp.get("compensation_stack", []),
            )
            await self._checkpoint_store.save(checkpoint)
            self.resources.create(
                "workflow_run_events", tenant_id, user_id,
                {"name": "checkpoint", "status": "saved", "workflow_id": workflow.id, "run_id": run.id, "output_payload": {"checkpoint_id": checkpoint.id}},
            )

        async def event_wait_cb(event_name: str, config: dict[str, Any]) -> Any:
            action = config.get("action", "wait")
            if action == "emit":
                self._event_bus.trigger(event_name, payload=config.get("payload", {}), source_run_id=run.id)
                return {"status": "emitted", "event_name": event_name}
            sub = self._event_bus.subscribe(
                workflow_run_id=run.id, node_id=config.get("node_id", ""),
                event_name=event_name, timeout_seconds=float(config.get("timeout_seconds", 3600)),
            )
            self.resources.create(
                "workflow_run_events", tenant_id, user_id,
                {"name": f"waiting_{event_name}", "status": "waiting", "workflow_id": workflow.id, "run_id": run.id, "output_payload": {"subscription_id": sub.id}},
            )
            result = await self._event_bus.wait_for_event(sub.id)
            return result

        async def compensation_cb(comp_config: dict[str, Any], ctx: dict[str, Any]) -> None:
            comp_type = comp_config.get("type", "")
            if comp_type == "http":
                async with httpx.AsyncClient(timeout=30) as client:
                    await client.request(comp_config.get("method", "POST"), comp_config.get("url", ""), json=ctx)
            elif comp_type == "agent":
                from app.services.agent_runner_service import AgentRunnerService
                await AgentRunnerService(self.db).run(tenant_id, user_id, comp_config.get("agent_id", ""), {"prompt": comp_config.get("prompt", "rollback"), "context": ctx})

        try:
            result = await self.executor.run(
                nodes, edges, handlers, initial_context=context,
                checkpoint_callback=checkpoint_cb,
                event_wait_callback=event_wait_cb,
                compensation_callback=compensation_cb,
                timeout_seconds=timeout_seconds,
            )
            status = result.get("status", "success")
            if status == "waiting_approval":
                self.resources.update(
                    "workflow_runs", tenant_id, user_id, run.id,
                    {"status": status, "output_payload": result},
                )
                waiting_info = {"approval_id": result.get("pending_approval", "")}
                for nid, nr in result.get("results", {}).items():
                    if isinstance(nr, dict) and nr.get("output") and isinstance(nr["output"], dict) and nr["output"].get("status") == "waiting_approval":
                        waiting_info = nr["output"]
                        break
                return {"run_id": run.id, "status": status, "waiting": waiting_info, "results": self._unwrap_results(result.get("results", {}))}
            self.resources.update(
                "workflow_runs", tenant_id, user_id, run.id,
                {"status": status, "finished_at": datetime.now(timezone.utc), "output_payload": result},
            )
            unwrapped = self._unwrap_results(result.get("results", {}))
            return {"run_id": run.id, "status": status, "results": unwrapped, "context": result.get("context", {})}
        except Exception as exc:
            self.resources.update(
                "workflow_runs", tenant_id, user_id, run.id,
                {"status": "failed", "finished_at": datetime.now(timezone.utc), "error_message": str(exc)},
            )
            self.resources.create(
                "workflow_run_events", tenant_id, user_id,
                {"name": "workflow failed", "status": "failed", "workflow_id": workflow.id, "run_id": run.id, "error_message": str(exc)},
            )
            raise

    async def resume_from_checkpoint(self, tenant_id: str, user_id: str, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Resume a workflow from its latest checkpoint."""
        run = self.resources.get("workflow_runs", tenant_id, run_id)
        if not run.workflow_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "run has no workflow_id", 422)

        checkpoint_id = payload.get("checkpoint_id")
        if checkpoint_id:
            checkpoint = await self._checkpoint_store.load(run_id, checkpoint_id)
        else:
            checkpoint = await self._checkpoint_store.load_latest(run_id)

        if not checkpoint:
            raise AppError(ErrorCode.NOT_FOUND, "no checkpoint found for this run", 404)

        workflow = self.resources.get("workflow_definitions", tenant_id, run.workflow_id)
        nodes = (run.spec or {}).get("nodes") or self._nodes(workflow, payload)
        edges = (run.spec or {}).get("edges") or self._edges(workflow, payload)
        handlers = self._build_handlers(tenant_id, user_id, run.workflow_id, run_id)

        self.resources.update("workflow_runs", tenant_id, user_id, run_id, {"status": "running"})

        try:
            result = await self.executor.resume(
                nodes, edges, handlers, checkpoint.to_dict(),
                timeout_seconds=float(payload.get("timeout_seconds", 0)),
            )
            status = result.get("status", "success")
            self.resources.update(
                "workflow_runs", tenant_id, user_id, run_id,
                {"status": status, "finished_at": datetime.now(timezone.utc), "output_payload": result},
            )
            return {"run_id": run_id, **result}
        except Exception as exc:
            self.resources.update(
                "workflow_runs", tenant_id, user_id, run_id,
                {"status": "failed", "finished_at": datetime.now(timezone.utc), "error_message": str(exc)},
            )
            raise

    async def trigger_event(self, tenant_id: str, user_id: str, event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Trigger an event that may resume waiting workflows."""
        notified = self._event_bus.trigger(event_name, payload=payload)
        self.resources.create(
            "workflow_run_events", tenant_id, user_id,
            {"name": f"event_{event_name}", "status": "triggered", "output_payload": {"event_name": event_name, "notified_subscriptions": notified}},
        )
        return {"event_name": event_name, "notified": notified}

    async def list_checkpoints(self, tenant_id: str, run_id: str) -> list[dict[str, Any]]:
        """List all checkpoints for a workflow run."""
        checkpoints = await self._checkpoint_store.list_checkpoints(run_id)
        return [cp.to_dict() for cp in checkpoints]

    def _build_handlers(self, tenant_id: str, user_id: str, workflow_id: str, run_id: str) -> dict[str, Any]:
        """Build node type handlers for the executor.

        Note: The executor passes node config (not the full node) as the first argument.
        """
        async def model_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return await self._invoke_model(tenant_id, user_id, config, context)

        async def rag_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return self._invoke_rag(tenant_id, user_id, config, context)

        async def agent_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return await self._invoke_agent(tenant_id, user_id, config, context)

        async def tool_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return await self._invoke_tool(tenant_id, user_id, config, context)

        async def http_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return await self._invoke_http(config, context)

        async def transform_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return self._transform(config, context)

        async def approval_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            approval = self.resources.create(
                "workflow_approvals", tenant_id, user_id,
                {"name": "approval", "status": "pending", "workflow_id": workflow_id, "run_id": run_id, "spec": {"approvers": config.get("approvers", [])}},
            )
            return {"status": "waiting_approval", "approval_id": approval.id}

        async def sub_workflow_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            sub_workflow_id = str(config.get("workflow_id", ""))
            sub_payload = dict(config.get("payload") or {})
            sub_payload["context"] = context
            return await self.run(tenant_id, user_id, sub_workflow_id, sub_payload)

        async def default_handler(config: dict[str, Any], context: dict[str, Any]) -> Any:
            return {"status": "success"}

        return {
            "model": model_handler, "chat_llm": model_handler,
            "vision_language": model_handler, "embedding": model_handler, "rerank": model_handler,
            "rag": rag_handler, "rag_retrieve": rag_handler,
            "agent": agent_handler,
            "tool": tool_handler,
            "http": http_handler,
            "transform": transform_handler, "script": transform_handler,
            "approval": approval_handler, "human_approval": approval_handler,
            "sub_workflow": sub_workflow_handler,
            "default": default_handler,
        }

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

    async def _invoke_model(self, tenant_id: str, user_id: str, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        model_type = str(config.get("model_type") or "chat_llm")
        model_id = str(config.get("model_id") or "")
        if not model_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "model node requires model_id", 422)
        invocation = await ModelInvocationService(self.db).invoke(tenant_id, user_id, model_id, model_type, dict(config.get("payload") or context))
        return {"status": "success", "type": "model", "invocation": invocation}

    def _invoke_rag(self, tenant_id: str, user_id: str, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        kb_id = str(config.get("knowledge_base_id") or "")
        if not kb_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "rag node requires knowledge_base_id", 422)
        retrieval = KnowledgeService(self.db).retrieve(tenant_id, user_id, kb_id, dict(config.get("payload") or context))
        return {"status": "success", "type": "rag", "retrieval": retrieval}

    async def _invoke_agent(self, tenant_id: str, user_id: str, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(config.get("agent_id") or "")
        if not agent_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "agent node requires agent_id", 422)
        from app.runtime.service import RuntimeControlService
        prompt = str(config.get("prompt") or context.get("input") or context.get("prompt") or str(context))
        result = await RuntimeControlService(self.db).debug_run(tenant_id, user_id, agent_id, {"prompt": prompt, **dict(config.get("runtime") or {})})
        return {"status": "success", "type": "agent", "agent": result}

    async def _invoke_tool(self, tenant_id: str, user_id: str, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_id = str(config.get("tool_id") or config.get("id") or "")
        if not tool_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "tool node requires tool_id", 422)
        result = await ToolService(self.db).invoke(tenant_id, user_id, tool_id, {"arguments": dict(config.get("arguments") or config.get("payload") or context)})
        return {"status": "success", "type": "tool", "tool": result}

    async def _invoke_http(self, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
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
    def _unwrap_results(results: dict[str, Any]) -> dict[str, Any]:
        """Unwrap executor result format to backward-compatible format.

        Executor returns: {node_id: {"status": ..., "output": ..., "error": ..., "duration_ms": ...}}
        We return: {node_id: output} for backward compatibility.
        """
        unwrapped: dict[str, Any] = {}
        for node_id, result in results.items():
            if isinstance(result, dict) and "output" in result:
                unwrapped[node_id] = result["output"]
            else:
                unwrapped[node_id] = result
        return unwrapped

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
