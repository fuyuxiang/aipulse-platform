from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.runtime.service import RuntimeControlService
from app.services.settings.guardrail_service import GuardrailService
from app.services.build.knowledge_service import KnowledgeService
from app.services.build.memory_service import MemoryService
from app.services.settings.model_services import ModelInvocationService
from app.services._shared.resource_service import ResourceService
from app.services.build.tool_service import ToolService


class AgentRunnerService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    async def run(self, tenant_id: str, user_id: str, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        agent = self.resources.get("agents", tenant_id, agent_id)
        if not agent.enabled:
            raise AppError(ErrorCode.BUSINESS_ERROR, "agent disabled", 409)
        session_id = str(payload.get("session_id") or f"agent-run-{uuid.uuid4().hex[:8]}")
        prompt = str(payload.get("prompt") or payload.get("content") or payload.get("message") or "")
        if not prompt.strip():
            raise AppError(ErrorCode.VALIDATION_ERROR, "agent prompt is required", 422)

        config = self._agent_config(agent, payload)
        run_context: dict[str, Any] = {
            "agent": ResourceService.to_dict(agent),
            "session_id": session_id,
            "prompt": prompt,
            "rag": {"sources": [], "context_text": ""},
            "memory": {"items": [], "context_text": ""},
            "tool_calls": [],
            "guardrails": {},
        }

        try:
            input_check = GuardrailService(self.db).check_input(
                tenant_id,
                user_id,
                {"content": prompt, "policy_ids": self._list(config, "guardrail_policy_ids"), "agent_id": agent_id},
            )
            run_context["guardrails"]["input"] = input_check
            if not input_check.get("passed", True):
                raise AppError(ErrorCode.FORBIDDEN, "agent input blocked by guardrails", 403)
            prompt = str(input_check.get("masked_content") or prompt)

            rag_context = self._build_rag_context(tenant_id, user_id, config, prompt)
            memory_context = self._build_memory_context(tenant_id, user_id, agent_id, session_id, config, prompt)
            run_context["rag"] = rag_context
            run_context["memory"] = memory_context

            messages = self._build_messages(agent, config, prompt, payload, rag_context, memory_context)
            model_id = str(payload.get("model_id") or config.get("model_id") or "")
            if not model_id:
                model_id = self._default_model_id(tenant_id, str(config.get("model_type") or agent.model_type or "chat_llm"))

            if model_id:
                execution = await self._run_with_model(tenant_id, user_id, agent_id, model_id, config, messages)
            else:
                execution = await self._run_with_runtime(tenant_id, user_id, agent_id, session_id, config, prompt, rag_context, memory_context)

            response = str(execution.get("response") or "")
            output_check = GuardrailService(self.db).check_output(
                tenant_id,
                user_id,
                {
                    "content": response,
                    "policy_ids": self._list(config, "guardrail_policy_ids"),
                    "agent_id": agent_id,
                    "knowledge_context": rag_context.get("sources", []),
                },
            )
            run_context["guardrails"]["output"] = output_check
            if not output_check.get("passed", True):
                raise AppError(ErrorCode.FORBIDDEN, "agent output blocked by guardrails", 403)
            response = str(output_check.get("masked_content") or response)

            memory_write = MemoryService(self.db).record_interaction(
                tenant_id,
                user_id,
                {
                    "prompt": prompt,
                    "response": response,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "subject_user_id": payload.get("subject_user_id") or payload.get("user_id") or user_id,
                    "memory_policy": config.get("memory_policy") or {},
                    "scope": (config.get("memory_policy") or {}).get("write_scope", "session"),
                    "shared": (config.get("memory_policy") or {}).get("shared", False),
                    "source": "agent_run",
                },
            )

            latency_ms = int((time.perf_counter() - started) * 1000)
            run = self.resources.create(
                "agent_run_records",
                tenant_id,
                user_id,
                {
                    "name": f"run {agent.name}",
                    "status": "success",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "model_id": model_id,
                    "model_type": str(config.get("model_type") or agent.model_type or "chat_llm"),
                    "latency_ms": latency_ms,
                    "token_usage": execution.get("token_usage", {}),
                    "cost": float(execution.get("cost") or 0),
                    "input_payload": {"prompt": prompt, "payload": payload},
                    "output_payload": {
                        "response": response,
                        "rag": rag_context,
                        "memory": memory_context,
                        "memory_write": memory_write,
                        "tool_calls": execution.get("tool_calls", []),
                        "guardrails": run_context["guardrails"],
                    },
                    "started_at": datetime.now(timezone.utc),
                    "finished_at": datetime.now(timezone.utc),
                },
            )
            return {
                "run_id": run.id,
                "agent_id": agent_id,
                "session_id": session_id,
                "status": "success",
                "response": response,
                "model_id": model_id,
                "latency_ms": latency_ms,
                "token_usage": execution.get("token_usage", {}),
                "cost": float(execution.get("cost") or 0),
                "rag": rag_context,
                "memory": memory_context,
                "memory_write": memory_write,
                "tool_calls": execution.get("tool_calls", []),
                "guardrails": run_context["guardrails"],
            }
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            self.resources.create(
                "agent_run_records",
                tenant_id,
                user_id,
                {
                    "name": f"run {agent.name}",
                    "status": "failed",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "latency_ms": latency_ms,
                    "input_payload": {"prompt": prompt, "payload": payload},
                    "output_payload": run_context,
                    "error_message": str(exc),
                    "started_at": datetime.now(timezone.utc),
                    "finished_at": datetime.now(timezone.utc),
                },
            )
            raise

    def _agent_config(self, agent: Any, payload: dict[str, Any]) -> dict[str, Any]:
        config = {**dict(agent.spec or {}), **dict(agent.config or {})}
        for key in ("model_id", "model_type", "system_prompt", "knowledge_base_ids", "tool_ids", "guardrail_policy_ids", "memory_policy", "rag_policy"):
            if key in payload:
                config[key] = payload[key]
        return config

    def _build_rag_context(self, tenant_id: str, user_id: str, config: dict[str, Any], prompt: str) -> dict[str, Any]:
        rag_policy = dict(config.get("rag_policy") or {})
        if rag_policy.get("enabled") is False:
            return {"sources": [], "retrieval_log_ids": [], "context_text": "", "total": 0}
        kb_ids = self._list(config, "knowledge_base_ids")
        if not kb_ids:
            return {"sources": [], "retrieval_log_ids": [], "context_text": "", "total": 0}
        return KnowledgeService(self.db).build_context(tenant_id, user_id, kb_ids, prompt, rag_policy)

    def _build_memory_context(self, tenant_id: str, user_id: str, agent_id: str, session_id: str, config: dict[str, Any], prompt: str) -> dict[str, Any]:
        memory_policy = dict(config.get("memory_policy") or {})
        if memory_policy.get("disabled") or memory_policy.get("enabled") is False:
            return {"items": [], "context_text": "", "total": 0}
        return MemoryService(self.db).build_context(
            tenant_id,
            user_id,
            {
                "query": prompt,
                "agent_id": agent_id,
                "session_id": session_id,
                "subject_user_id": user_id,
                "team_id": memory_policy.get("team_id", ""),
                "include_shared": memory_policy.get("include_shared", True),
                "top_k": int(memory_policy.get("top_k") or 8),
            },
        )

    def _build_messages(self, agent: Any, config: dict[str, Any], prompt: str, payload: dict[str, Any], rag_context: dict[str, Any], memory_context: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system_prompt = str(config.get("system_prompt") or (agent.spec or {}).get("system_prompt") or "")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if memory_context.get("context_text"):
            messages.append({"role": "system", "content": f"Shared memory context:\n{memory_context['context_text']}"})
        if rag_context.get("context_text"):
            messages.append({"role": "system", "content": f"RAG knowledge context:\n{rag_context['context_text']}"})
        context = payload.get("context") or []
        if isinstance(context, list):
            for item in context[-20:]:
                if isinstance(item, dict) and item.get("content"):
                    messages.append({"role": str(item.get("role") or "user"), "content": str(item.get("content"))})
        elif isinstance(context, dict):
            messages.append({"role": "system", "content": f"Runtime context:\n{json.dumps(context, ensure_ascii=False)}"})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _run_with_model(self, tenant_id: str, user_id: str, agent_id: str, model_id: str, config: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
        model_type = str(config.get("model_type") or "chat_llm")
        tools = self._model_tools(tenant_id, self._list(config, "tool_ids"))
        invocation = await ModelInvocationService(self.db).invoke(
            tenant_id,
            user_id,
            model_id,
            model_type,
            {
                "agent_id": agent_id,
                "messages": messages,
                "temperature": config.get("temperature", 0.7),
                "max_tokens": config.get("max_tokens", 4096),
                "tools": [tool["definition"] for tool in tools],
            },
        )
        result = dict(invocation.get("result") or {})
        tool_calls = await self._execute_tool_calls(tenant_id, user_id, tools, result.get("tool_calls") or [])
        if tool_calls:
            messages.append({"role": "assistant", "content": result.get("content") or "", "tool_calls": result.get("tool_calls") or []})
            for call in tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(call.get("output", {}), ensure_ascii=False)})
            invocation = await ModelInvocationService(self.db).invoke(
                tenant_id,
                user_id,
                model_id,
                model_type,
                {
                    "agent_id": agent_id,
                    "messages": messages,
                    "temperature": config.get("temperature", 0.7),
                    "max_tokens": config.get("max_tokens", 4096),
                },
            )
            result = dict(invocation.get("result") or {})
        return {
            "response": result.get("content", ""),
            "token_usage": result.get("usage", {}),
            "cost": invocation.get("cost", 0),
            "tool_calls": tool_calls,
        }

    async def _run_with_runtime(self, tenant_id: str, user_id: str, agent_id: str, session_id: str, config: dict[str, Any], prompt: str, rag_context: dict[str, Any], memory_context: dict[str, Any]) -> dict[str, Any]:
        augmented_prompt = "\n\n".join(
            part
            for part in [
                f"Shared memory context:\n{memory_context.get('context_text')}" if memory_context.get("context_text") else "",
                f"RAG knowledge context:\n{rag_context.get('context_text')}" if rag_context.get("context_text") else "",
                f"User request:\n{prompt}",
            ]
            if part
        )
        result = await RuntimeControlService(self.db).debug_run(
            tenant_id,
            user_id,
            agent_id,
            {
                "prompt": augmented_prompt,
                "session_id": session_id,
                "model_config": config.get("model_config") or {},
                "tool_policy": config.get("tool_policy") or {"allow": self._list(config, "tool_ids")},
                "memory_policy": config.get("memory_policy") or {},
                "knowledge_bindings": [{"knowledge_base_id": item} for item in self._list(config, "knowledge_base_ids")],
            },
        )
        return {"response": result.get("response", ""), "token_usage": {}, "cost": 0, "tool_calls": []}

    def _model_tools(self, tenant_id: str, tool_ids: list[str]) -> list[dict[str, Any]]:
        tools = []
        for tool_id in tool_ids:
            tool = self.resources.get("tools", tenant_id, tool_id)
            schema = dict((tool.spec or {}).get("schema") or (tool.config or {}).get("schema") or {"type": "object", "properties": {}})
            name = str(tool.code or tool.tool_name or tool.name or tool.id).replace(" ", "_")
            tools.append(
                {
                    "id": tool.id,
                    "name": name,
                    "definition": {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.description or tool.name,
                            "parameters": schema,
                        },
                    },
                }
            )
        return tools

    async def _execute_tool_calls(self, tenant_id: str, user_id: str, tools: list[dict[str, Any]], calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_by_name = {tool["name"]: tool for tool in tools}
        executed = []
        for call in calls:
            function = dict(call.get("function") or {})
            name = str(function.get("name") or call.get("name") or "")
            tool = tool_by_name.get(name)
            if not tool:
                continue
            arguments = function.get("arguments") or call.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            result = await ToolService(self.db).invoke(tenant_id, user_id, str(tool["id"]), {"arguments": dict(arguments), "approved": True})
            executed.append({"id": call.get("id", ""), "name": name, "tool_id": tool["id"], "arguments": arguments, "output": result.get("output", {})})
        return executed

    def _default_model_id(self, tenant_id: str, model_type: str) -> str:
        rows, _ = self.resources.list("models", tenant_id, 1, 100, {"status": "active"})
        for row in rows:
            if row.enabled and row.model_type == model_type:
                return row.id
        for row in rows:
            if row.enabled and row.model_type == "chat_llm":
                return row.id
        return ""

    @staticmethod
    def _list(config: dict[str, Any], key: str) -> list[str]:
        value = config.get(key) or []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item) for item in value if str(item)]
        return []
