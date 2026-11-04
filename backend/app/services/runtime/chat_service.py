from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.models.resources import RESOURCE_MODELS
from app.core.constants import ErrorCode
from app.core.errors import AppError

from app.services._shared.resource_service import ResourceService


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def create_session(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", ""))
        if agent_id:
            self.resources.get("agents", tenant_id, agent_id)
        title = str(payload.get("title", "新对话"))
        session = self.resources.create("chat_sessions", tenant_id, user_id, {
            "name": title,
            "code": f"chat-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": agent_id,
            "user_id": user_id,
            "spec": {
                "agent_id": agent_id,
                "model_id": str(payload.get("model_id") or ""),
                "title": title,
                "message_count": 0,
                "last_message_at": None,
                "context_window": int(payload.get("context_window", 20)),
                "temperature": float(payload.get("temperature", 0.7)),
                "max_tokens": int(payload.get("max_tokens", 4096)),
                "system_prompt": str(payload.get("system_prompt", "")),
                "knowledge_base_ids": payload.get("knowledge_base_ids", []),
                "tool_ids": payload.get("tool_ids", []),
                "memory_enabled": bool(payload.get("memory_enabled", True)),
                "guardrail_policy_ids": payload.get("guardrail_policy_ids", []),
            },
        })
        return ResourceService.to_dict(session)

    def list_sessions(self, tenant_id: str, user_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("chat_sessions", tenant_id, page, page_size, {"user_id": user_id, "status": "active"})
        return [ResourceService.to_dict(row) for row in rows], total

    def require_session(self, tenant_id: str, user_id: str, session_id: str) -> Any:
        session = self.resources.get("chat_sessions", tenant_id, session_id)
        if session.user_id != user_id:
            raise AppError(ErrorCode.NOT_FOUND, "chat session not found", 404)
        return session

    def get_session(self, tenant_id: str, session_id: str, user_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.require_session(tenant_id, user_id, session_id))

    def update_session(self, tenant_id: str, user_id: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_session(tenant_id, user_id, session_id)
        session = self.require_session(tenant_id, user_id, session_id)
        values = {k: v for k, v in payload.items() if k in {"name", "status"}}
        if isinstance(payload.get("spec"), dict):
            values["spec"] = {**session.spec, **{k: v for k, v in payload["spec"].items() if k in {"pinned", "title", "context_window", "temperature", "max_tokens"}}}
        row = self.resources.update("chat_sessions", tenant_id, user_id, session_id, values)
        return ResourceService.to_dict(row)

    def delete_session(self, tenant_id: str, user_id: str, session_id: str) -> dict[str, str]:
        self.require_session(tenant_id, user_id, session_id)
        return self.resources.delete("chat_sessions", tenant_id, user_id, session_id)

    def archive_session(self, tenant_id: str, user_id: str, session_id: str) -> dict[str, Any]:
        self.require_session(tenant_id, user_id, session_id)
        row = self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"status": "archived"})
        return ResourceService.to_dict(row)

    def pin_session(self, tenant_id: str, user_id: str, session_id: str, pinned: bool) -> dict[str, Any]:
        self.require_session(tenant_id, user_id, session_id)
        session = self.resources.get("chat_sessions", tenant_id, session_id)
        spec = dict(session.spec or {})
        spec["pinned"] = pinned
        row = self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"spec": spec})
        return ResourceService.to_dict(row)

    def send_message(self, tenant_id: str, user_id: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_session(tenant_id, user_id, session_id)
        content = str(payload.get("content", ""))
        role = str(payload.get("role", "user"))
        attachments = payload.get("attachments", [])
        parent_message_id = str(payload.get("parent_message_id", ""))
        msg = self.resources.create("chat_messages", tenant_id, user_id, {
            "name": content[:80] if content else "message",
            "code": f"msg-{uuid.uuid4().hex[:8]}",
            "status": "sent",
            "session_id": session_id,
            "agent_id": self._get_session_agent_id(tenant_id, session_id),
            "user_id": user_id,
            "parent_id": parent_message_id,
            "spec": {
                "role": role,
                "content": content,
                "attachments": attachments,
                "token_usage": {},
                "latency_ms": 0,
                "model_id": "",
                "feedback": None,
                "metadata": payload.get("metadata", {}),
            },
        })
        session = self.resources.get("chat_sessions", tenant_id, session_id)
        spec = dict(session.spec or {})
        spec["message_count"] = spec.get("message_count", 0) + 1
        spec["last_message_at"] = datetime.now(timezone.utc).isoformat()
        self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"spec": spec})
        return ResourceService.to_dict(msg)

    def list_messages(self, tenant_id: str, session_id: str, page: int, page_size: int, user_id: str) -> tuple[list[dict[str, Any]], int]:
        self.require_session(tenant_id, user_id, session_id)
        model: Any = RESOURCE_MODELS["chat_messages"]
        criteria = (model.tenant_id == tenant_id, model.session_id == session_id, model.deleted_at.is_(None))
        rows = list(self.db.scalars(select(model).where(*criteria).order_by(model.created_at.desc(), model.id.desc()).offset((page - 1) * page_size).limit(page_size)))
        total = int(self.db.scalar(select(func.count()).select_from(model).where(*criteria)) or 0)
        return [ResourceService.to_dict(row) for row in reversed(rows)], total

    def get_message(self, tenant_id: str, message_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("chat_messages", tenant_id, message_id))

    def feedback_message(self, tenant_id: str, user_id: str, message_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        msg = self.resources.get("chat_messages", tenant_id, message_id)
        self.require_session(tenant_id, user_id, msg.session_id)
        spec = dict(msg.spec or {})
        spec["feedback"] = {
            "rating": payload.get("rating"),
            "comment": str(payload.get("comment", "")),
            "tags": payload.get("tags", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        row = self.resources.update("chat_messages", tenant_id, user_id, message_id, {"spec": spec})
        if payload.get("rating") == "bad":
            self.resources.create("bad_cases", tenant_id, user_id, {
                "name": f"bad-case-{message_id[:8]}",
                "code": f"bc-{uuid.uuid4().hex[:8]}",
                "status": "pending",
                "session_id": msg.session_id,
                "agent_id": msg.agent_id,
                "parent_id": message_id,
                "spec": {
                    "message_content": spec.get("content", ""),
                    "feedback_comment": str(payload.get("comment", "")),
                    "tags": payload.get("tags", []),
                },
            })
        return ResourceService.to_dict(row)

    def regenerate_message(self, tenant_id: str, user_id: str, session_id: str, message_id: str) -> dict[str, Any]:
        self.require_session(tenant_id, user_id, session_id)
        msg = self.resources.get("chat_messages", tenant_id, message_id)
        if msg.session_id != session_id or (msg.spec or {}).get("role") != "assistant":
            raise AppError(ErrorCode.VALIDATION_ERROR, "target must be an assistant message in this session", 422)
        parent = self.resources.get("chat_messages", tenant_id, msg.parent_id) if msg.parent_id else None
        if parent is None:
            model: Any = RESOURCE_MODELS["chat_messages"]
            rows = self.db.scalars(select(model).where(model.tenant_id == tenant_id, model.session_id == session_id,
                model.created_at <= msg.created_at, model.id != msg.id, model.deleted_at.is_(None)).order_by(model.created_at.desc())).all()
            parent = next((r for r in rows if (r.spec or {}).get("role") == "user"), None)
        if parent is None or parent.session_id != session_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "original user message not found", 422)
        return {"content": (parent.spec or {}).get("content", ""), "_parent_id": parent.id, "_regenerate_id": msg.id}

    async def stream_reply(self, tenant_id: str, user_id: str, session_id: str, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        session = self.require_session(tenant_id, user_id, session_id)
        content = str(payload.get("content", "")).strip()
        if not content:
            raise AppError(ErrorCode.VALIDATION_ERROR, "message content is required", 422)
        spec = dict(session.spec or {})
        agent_id = str(spec.get("agent_id") or "")
        context = self._build_context(tenant_id, session_id, int(spec.get("context_window") or 20))
        if payload.get("_parent_id"):
            parent = self.resources.get("chat_messages", tenant_id, str(payload["_parent_id"]))
            # Only the history before the original request belongs in a regeneration.
            model: Any = RESOURCE_MODELS["chat_messages"]
            rows = self.db.scalars(select(model).where(model.tenant_id == tenant_id, model.session_id == session_id,
                model.created_at < parent.created_at, model.status == "sent", model.deleted_at.is_(None)).order_by(model.created_at.desc()).limit(int(spec.get("context_window") or 20))).all()
            context = [{"role": r.spec.get("role", "user"), "content": r.spec.get("content", "")} for r in reversed(rows)]
        else:
            parent = self.resources.get("chat_messages", tenant_id, self.send_message(tenant_id, user_id, session_id, {"content": content})["id"])
        started = time.perf_counter()
        reply = self.resources.create("chat_messages", tenant_id, user_id, {
            "name": "reply", "status": "running", "session_id": session_id, "agent_id": agent_id,
            "user_id": user_id, "parent_id": parent.id,
            "spec": {"role": "assistant", "content": "", "regenerates": payload.get("_regenerate_id", ""),
                     "delivery_mode": "buffered", "token_usage_source": "estimated"}})
        yield self._sse_event("message_start", {"message_id": reply.id, "agent_id": agent_id, "delivery_mode": "buffered"})
        try:
            execution = await self._call_agent(tenant_id, user_id, agent_id, session_id, content, context, [], list(spec.get("tool_ids") or []), spec)
            details = execution if isinstance(execution, dict) else {"response": execution}
            if details.get("status") == "waiting_approval":
                self.resources.update("chat_messages", tenant_id, user_id, reply.id, {"status": "waiting_approval",
                    "spec": {**reply.spec, "approval_id": details.get("approval_id"), "tool_calls": details.get("tool_calls", [])}})
                yield self._sse_event("approval_required", {"message_id": reply.id, "approval_id": details.get("approval_id")})
                return
            response = str(details.get("response") or "")
            sources = (details.get("rag") or {}).get("sources", [])
            if sources:
                yield self._sse_event("rag_context", {"sources": sources})
            reply_spec = {**reply.spec, "content": response, "rag_sources": sources, "tool_calls": details.get("tool_calls", []),
                          "model_id": details.get("model_id", ""),
                          "token_usage_source": "provider" if details.get("token_usage") else "estimated", "latency_ms": int((time.perf_counter() - started) * 1000),
                          "token_usage": details.get("token_usage") or self._estimate_tokens(content, response)}
            self.resources.update("chat_messages", tenant_id, user_id, reply.id, {"status": "sent", "spec": reply_spec})
            if payload.get("_regenerate_id"):
                self.resources.update("chat_messages", tenant_id, user_id, str(payload["_regenerate_id"]), {"status": "regenerated"})
            yield self._sse_event("content_delta", {"delta": response})
            yield self._sse_event("message_end", {"message_id": reply.id, "token_usage": reply_spec["token_usage"],
                "token_usage_source": reply_spec["token_usage_source"], "latency_ms": reply_spec["latency_ms"]})
        except asyncio.CancelledError:
            self.resources.update("chat_messages", tenant_id, user_id, reply.id, {"status": "cancelled"})
            raise
        except Exception as exc:
            self.resources.update("chat_messages", tenant_id, user_id, reply.id, {"status": "failed", "error_message": str(exc)})
            yield self._sse_event("error", {"message_id": reply.id, "error": str(exc)})
        finally:
            model = RESOURCE_MODELS["chat_messages"]
            count = self.db.scalar(select(func.count()).select_from(model).where(model.tenant_id == tenant_id,
                model.session_id == session_id, model.deleted_at.is_(None)))
            self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"spec": {**session.spec,
                "message_count": count, "last_message_at": datetime.now(timezone.utc).isoformat()}})

    async def _call_agent(
        self, tenant_id: str, user_id: str, agent_id: str, session_id: str,
        prompt: str, context: list[dict[str, str]], rag_context: list[dict[str, Any]],
        tool_ids: list[str], session_spec: dict[str, Any],
    ) -> dict[str, Any]:
        model_id = str(session_spec.get("model_id") or "") if not agent_id else ""
        if model_id:
            from app.services.settings.model_services import ModelInvocationService

            messages = []
            system_prompt = str(session_spec.get("system_prompt") or "")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if rag_context:
                rag_text = "\n".join(f"[{s.get('title', '')}] {s.get('content', '')}" for s in rag_context)
                messages.append({"role": "system", "content": f"Use the following retrieved knowledge when relevant:\n{rag_text}"})
            messages.extend(context[-int(session_spec.get("context_window", 20) or 20):])
            messages.append({"role": "user", "content": prompt})
            invocation = await ModelInvocationService(self.db).invoke(
                tenant_id,
                user_id,
                model_id,
                "chat_llm",
                {
                    "messages": messages,
                    "temperature": session_spec.get("temperature", 0.7),
                    "max_tokens": session_spec.get("max_tokens", 4096),
                    "tool_ids": tool_ids,
                },
            )
            result = invocation.get("result") or {}
            return {"response": result.get("content", ""), "token_usage": result.get("usage", {}), "model_id": model_id}
        if agent_id:
            from app.services.runtime.agent_runner_service import AgentRunnerService

            result = await AgentRunnerService(self.db).run(
                tenant_id,
                user_id,
                agent_id,
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "context": context,

                }
            )
            return result
        from app.core.constants import ErrorCode
        from app.core.errors import AppError

        raise AppError(ErrorCode.VALIDATION_ERROR, "chat session requires agent_id or model_id", 422)

    def _build_context(self, tenant_id: str, session_id: str, window: int) -> list[dict[str, str]]:
        rows, _ = self.resources.list("chat_messages", tenant_id, 1, window, {"session_id": session_id})
        messages = []
        for row in reversed(rows):
            if row.status != "sent":
                continue
            spec = row.spec or {}
            messages.append({"role": spec.get("role", "user"), "content": spec.get("content", "")})
        return messages

    def _get_session_agent_id(self, tenant_id: str, session_id: str) -> str:
        try:
            session = self.resources.get("chat_sessions", tenant_id, session_id)
            return (session.spec or {}).get("agent_id", session.agent_id or "")
        except Exception:
            return ""

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int = 20) -> list[str]:
        if not text:
            return [""]
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    @staticmethod
    def _estimate_tokens(prompt: str, response: str) -> dict[str, int]:
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(response) // 4)
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}

    @staticmethod
    def _sse_event(event_type: str, data: dict[str, Any]) -> str:
        import json
        payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
        return f"event: {event_type}\ndata: {payload}\n\n"
