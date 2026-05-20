from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services._shared.resource_service import ResourceService


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def create_session(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", ""))
        title = str(payload.get("title", "新对话"))
        session = self.resources.create("chat_sessions", tenant_id, user_id, {
            "name": title,
            "code": f"chat-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": agent_id,
            "user_id": user_id,
            "spec": {
                "agent_id": agent_id,
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

    def get_session(self, tenant_id: str, session_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("chat_sessions", tenant_id, session_id))

    def update_session(self, tenant_id: str, user_id: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("chat_sessions", tenant_id, user_id, session_id, payload)
        return ResourceService.to_dict(row)

    def delete_session(self, tenant_id: str, user_id: str, session_id: str) -> dict[str, str]:
        return self.resources.delete("chat_sessions", tenant_id, user_id, session_id)

    def archive_session(self, tenant_id: str, user_id: str, session_id: str) -> dict[str, Any]:
        row = self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"status": "archived"})
        return ResourceService.to_dict(row)

    def pin_session(self, tenant_id: str, user_id: str, session_id: str, pinned: bool) -> dict[str, Any]:
        session = self.resources.get("chat_sessions", tenant_id, session_id)
        spec = dict(session.spec or {})
        spec["pinned"] = pinned
        row = self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"spec": spec})
        return ResourceService.to_dict(row)

    def send_message(self, tenant_id: str, user_id: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
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

    def list_messages(self, tenant_id: str, session_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("chat_messages", tenant_id, page, page_size, {"session_id": session_id})
        return [ResourceService.to_dict(row) for row in rows], total

    def get_message(self, tenant_id: str, message_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("chat_messages", tenant_id, message_id))

    def feedback_message(self, tenant_id: str, user_id: str, message_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        msg = self.resources.get("chat_messages", tenant_id, message_id)
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
        msg = self.resources.get("chat_messages", tenant_id, message_id)
        spec = dict(msg.spec or {})
        spec["status"] = "regenerated"
        self.resources.update("chat_messages", tenant_id, user_id, message_id, {"status": "regenerated", "spec": spec})
        return {"session_id": session_id, "original_message_id": message_id, "status": "regenerating"}

    async def stream_reply(self, tenant_id: str, user_id: str, session_id: str, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        content = str(payload.get("content", ""))
        self.send_message(tenant_id, user_id, session_id, {"content": content, "role": "user"})

        session = self.resources.get("chat_sessions", tenant_id, session_id)
        spec = dict(session.spec or {})
        agent_id = spec.get("agent_id", "")
        context_window = spec.get("context_window", 20)
        knowledge_base_ids = spec.get("knowledge_base_ids", [])
        tool_ids = spec.get("tool_ids", [])

        context_messages = self._build_context(tenant_id, session_id, context_window)
        rag_context = self._retrieve_knowledge(tenant_id, user_id, knowledge_base_ids, content) if knowledge_base_ids else []

        start_time = time.time()
        reply_id = f"msg-{uuid.uuid4().hex[:8]}"

        yield self._sse_event("message_start", {"message_id": reply_id, "agent_id": agent_id})

        if rag_context:
            yield self._sse_event("rag_context", {"sources": rag_context})

        try:
            full_response = await self._call_agent(tenant_id, user_id, agent_id, session_id, content, context_messages, rag_context, tool_ids, spec)
        except Exception as exc:
            latency_ms = int((time.time() - start_time) * 1000)
            failed = self.resources.create("chat_messages", tenant_id, user_id, {
                "name": f"failed reply {reply_id}",
                "code": reply_id,
                "status": "failed",
                "session_id": session_id,
                "agent_id": agent_id,
                "user_id": "assistant",
                "error_message": str(exc),
                "spec": {
                    "role": "assistant",
                    "content": "",
                    "attachments": [],
                    "token_usage": {},
                    "latency_ms": latency_ms,
                    "model_id": spec.get("model_id", ""),
                    "feedback": None,
                    "rag_sources": rag_context,
                    "tool_calls": [],
                    "metadata": {"error": str(exc)},
                },
            })
            yield self._sse_event("error", {"message_id": failed.id, "error": str(exc), "latency_ms": latency_ms})
            return

        chunks = self._split_into_chunks(full_response)
        for i, chunk in enumerate(chunks):
            yield self._sse_event("content_delta", {"index": i, "delta": chunk})
            await asyncio.sleep(0.02)

        latency_ms = int((time.time() - start_time) * 1000)
        token_usage = self._estimate_tokens(content, full_response)

        assistant_msg = self.resources.create("chat_messages", tenant_id, user_id, {
            "name": full_response[:80] if full_response else "reply",
            "code": reply_id,
            "status": "sent",
            "session_id": session_id,
            "agent_id": agent_id,
            "user_id": "assistant",
            "spec": {
                "role": "assistant",
                "content": full_response,
                "attachments": [],
                "token_usage": token_usage,
                "latency_ms": latency_ms,
                "model_id": spec.get("model_id", ""),
                "feedback": None,
                "rag_sources": rag_context,
                "tool_calls": [],
                "thinking": "",
                "metadata": {},
            },
        })

        session_spec = dict(session.spec or {})
        session_spec["message_count"] = session_spec.get("message_count", 0) + 2
        session_spec["last_message_at"] = datetime.now(timezone.utc).isoformat()
        self.resources.update("chat_sessions", tenant_id, user_id, session_id, {"spec": session_spec})

        yield self._sse_event("message_end", {
            "message_id": assistant_msg.id,
            "token_usage": token_usage,
            "latency_ms": latency_ms,
            "model_id": spec.get("model_id", ""),
        })

    async def _call_agent(
        self, tenant_id: str, user_id: str, agent_id: str, session_id: str,
        prompt: str, context: list[dict[str, str]], rag_context: list[dict[str, Any]],
        tool_ids: list[str], session_spec: dict[str, Any],
    ) -> str:
        model_id = str(session_spec.get("model_id") or "")
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
            return str((invocation.get("result") or {}).get("content") or "")
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
                    "knowledge_base_ids": session_spec.get("knowledge_base_ids", []),
                    "tool_ids": session_spec.get("tool_ids", []),
                    "guardrail_policy_ids": session_spec.get("guardrail_policy_ids", []),
                    "memory_policy": {"enabled": session_spec.get("memory_enabled", True), "write_scope": "session", "include_shared": True},
                }
            )
            return str(result.get("response", ""))
        from app.core.constants import ErrorCode
        from app.core.errors import AppError

        raise AppError(ErrorCode.VALIDATION_ERROR, "chat session requires agent_id or model_id", 422)

    def _build_context(self, tenant_id: str, session_id: str, window: int) -> list[dict[str, str]]:
        rows, _ = self.resources.list("chat_messages", tenant_id, 1, window, {"session_id": session_id})
        messages = []
        for row in rows:
            spec = row.spec or {}
            messages.append({"role": spec.get("role", "user"), "content": spec.get("content", "")})
        return messages

    def _retrieve_knowledge(self, tenant_id: str, user_id: str, kb_ids: list[str], query: str) -> list[dict[str, Any]]:
        results = []
        try:
            from app.services.build.knowledge_service import KnowledgeService
            ks = KnowledgeService(self.db)
            for kb_id in kb_ids[:3]:
                try:
                    result = ks.retrieve(tenant_id, user_id, kb_id, {"query": query, "top_k": 3})
                    for chunk in result.get("matches", []) + result.get("chunks", []):
                        metadata = dict(chunk.get("metadata") or {})
                        results.append({
                            "knowledge_base_id": kb_id,
                            "chunk_id": chunk.get("id", ""),
                            "title": chunk.get("title") or metadata.get("document_name") or metadata.get("filename") or "",
                            "content": chunk.get("content") or chunk.get("text", ""),
                            "score": chunk.get("score", 0),
                        })
                except Exception:
                    pass
        except ImportError:
            pass
        return results

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
