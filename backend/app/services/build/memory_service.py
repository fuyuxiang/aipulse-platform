from __future__ import annotations

import re
import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services._shared.resource_service import ResourceService
from app.services.settings.security_service import SecurityService


class MemoryService:
    def __init__(self, db: Session):
        self.resources = ResourceService(db)

    def search(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or payload.get("text") or "")
        scope = str(payload.get("scope") or "")
        agent_id = str(payload.get("agent_id") or "")
        session_id = str(payload.get("session_id") or "")
        subject_user_id = str(payload.get("subject_user_id") or payload.get("user_id") or user_id)
        team_id = str(payload.get("team_id") or "")
        include_shared = bool(payload.get("include_shared", True))
        rows, _ = self.resources.list("memory_items", tenant_id, 1, int(payload.get("limit") or 200))
        matches = []
        for row in rows:
            spec = dict(row.spec or {})
            if row.status != "active":
                continue
            if scope and spec.get("scope") != scope:
                continue
            if not self._visible(row, spec, subject_user_id, agent_id, session_id, team_id, include_shared):
                continue
            score = self._score(query, f"{row.name} {row.description} {row.spec}")
            if not query or score > 0 or spec.get("scope") in {"tenant", "global", "shared"} or spec.get("always_include"):
                item = ResourceService.to_dict(row)
                item["relevance_score"] = score
                matches.append(item)
        matches.sort(key=lambda item: float(item.get("relevance_score") or 0), reverse=True)
        operation = self.resources.create(
            "memory_operations",
            tenant_id,
            user_id,
            {"name": "memory search", "status": "completed", "input_payload": payload, "output_payload": {"count": len(matches)}},
        )
        return {"operation_id": operation.id, "total": len(matches), "items": matches}

    def remember(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        content = str(payload.get("content") or payload.get("text") or payload.get("description") or "").strip()
        if not content:
            from app.core.constants import ErrorCode
            from app.core.errors import AppError

            raise AppError(ErrorCode.VALIDATION_ERROR, "memory content is required", 422)
        scope = str(payload.get("scope") or "session")
        agent_id = str(payload.get("agent_id") or "")
        session_id = str(payload.get("session_id") or "")
        subject_user_id = str(payload.get("subject_user_id") or payload.get("user_id") or user_id)
        team_id = str(payload.get("team_id") or "")
        content_hash = self._hash(content, scope, agent_id, session_id, subject_user_id, team_id)
        existing = self._find_by_hash(tenant_id, content_hash)
        spec = {
            "scope": scope,
            "form": str(payload.get("form") or "episodic"),
            "shared": bool(payload.get("shared") or scope in {"tenant", "global", "shared", "team"}),
            "source": str(payload.get("source") or "manual"),
            "confidence": float(payload.get("confidence") or 0.8),
            "hash": content_hash,
            "agent_id": agent_id,
            "session_id": session_id,
            "subject_user_id": subject_user_id,
            "team_id": team_id,
            "tags": payload.get("tags", []),
            "metadata": payload.get("metadata", {}),
            "last_reinforced_at": datetime.now(timezone.utc).isoformat(),
            "reinforcement_count": 1,
        }
        if existing is not None:
            old_spec = dict(existing.spec or {})
            old_spec.update(spec)
            old_spec["reinforcement_count"] = int((existing.spec or {}).get("reinforcement_count") or 1) + 1
            row = self.resources.update(
                "memory_items",
                tenant_id,
                user_id,
                existing.id,
                {
                    "description": content,
                    "agent_id": agent_id or existing.agent_id,
                    "session_id": session_id or existing.session_id,
                    "user_id": subject_user_id,
                    "spec": old_spec,
                },
            )
            action = "reinforce"
        else:
            row = self.resources.create(
                "memory_items",
                tenant_id,
                user_id,
                {
                    "name": str(payload.get("name") or content[:120]),
                    "description": content,
                    "status": "active",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "user_id": subject_user_id,
                    "parent_id": team_id,
                    "spec": spec,
                },
            )
            action = "create"
        self._audit(tenant_id, user_id, row.id, action, {"scope": scope, "shared": spec["shared"], "source": spec["source"]})
        return {"memory_id": row.id, "status": action, "memory": ResourceService.to_dict(row)}

    def build_context(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        search = self.search(tenant_id, user_id, {**payload, "include_shared": payload.get("include_shared", True)})
        limit = int(payload.get("top_k") or payload.get("limit") or 8)
        items = search["items"][:limit]
        lines = []
        for index, item in enumerate(items, 1):
            spec = dict(item.get("spec") or {})
            scope = spec.get("scope", "memory")
            lines.append(f"[M{index}][{scope}] {item.get('description') or item.get('name')}")
        operation = self.resources.create(
            "memory_operations",
            tenant_id,
            user_id,
            {
                "name": "memory context",
                "status": "completed",
                "input_payload": payload,
                "output_payload": {"count": len(items), "memory_ids": [item["id"] for item in items]},
            },
        )
        return {"operation_id": operation.id, "items": items, "total": len(items), "context_text": "\n".join(lines)}

    def record_interaction(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        policy = dict(payload.get("memory_policy") or {})
        if policy.get("disabled") or policy.get("enabled") is False:
            return {"status": "disabled", "memory_ids": []}
        scope = str(payload.get("scope") or policy.get("write_scope") or "session")
        shared = bool(payload.get("shared") or policy.get("shared") or scope in {"tenant", "global", "shared", "team"})
        prompt = str(payload.get("prompt") or "")
        response = str(payload.get("response") or "")
        source = str(payload.get("source") or "agent_run")
        created: list[str] = []
        if policy.get("store_interactions", True):
            text = f"User: {prompt}\nAssistant: {response}".strip()
            if text:
                created.append(self.remember(tenant_id, user_id, {**payload, "content": text, "scope": scope, "shared": shared, "source": source, "form": "episodic"})["memory_id"])
        if policy.get("auto_extract", True):
            extracted = self.extract(
                tenant_id,
                user_id,
                {
                    **payload,
                    "text": f"{prompt}\n{response}",
                    "scope": str(policy.get("extract_scope") or ("tenant" if shared else "user")),
                    "shared": shared,
                    "source": f"{source}:extract",
                    "form": "semantic",
                    "confidence": policy.get("confidence", 0.7),
                },
            )
            created.extend(extracted.get("memory_ids", []))
        return {"status": "completed", "memory_ids": created, "count": len(created)}

    def extract(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or "")
        scope = str(payload.get("scope") or "session")
        form = str(payload.get("form") or "episodic")
        facts = [part.strip(" .") for part in re.split(r"[。\n.;]+", text) if len(part.strip()) > 2]
        created = []
        for fact in facts:
            remembered = self.remember(tenant_id, user_id, {**payload, "content": fact, "scope": scope, "form": form, "source": str(payload.get("source") or "extraction")})
            created.append(remembered["memory_id"])
        job = self.resources.create(
            "memory_extraction_jobs",
            tenant_id,
            user_id,
            {"name": "memory extraction", "status": "completed", "input_payload": payload, "output_payload": {"memory_ids": created}},
        )
        return {"job_id": job.id, "memory_ids": created, "count": len(created)}

    def merge(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        memory_ids = [str(item) for item in payload.get("memory_ids", [])]
        rows = [self.resources.get("memory_items", tenant_id, memory_id) for memory_id in memory_ids]
        merged_text = "\n".join(row.description or row.name for row in rows)
        merged = self.resources.create(
            "memory_items",
            tenant_id,
            user_id,
            {
                "name": str(payload.get("name") or "merged memory"),
                "description": merged_text,
                "status": "active",
                "spec": {"scope": payload.get("scope", "agent"), "form": payload.get("form", "semantic"), "source_ids": memory_ids},
            },
        )
        job = self.resources.create("memory_merge_jobs", tenant_id, user_id, {"name": "memory merge", "status": "completed", "output_payload": {"memory_id": merged.id}})
        return {"job_id": job.id, "memory_id": merged.id, "source_ids": memory_ids}

    def archive(self, tenant_id: str, user_id: str, memory_id: str) -> dict[str, Any]:
        row = self.resources.update("memory_items", tenant_id, user_id, memory_id, {"status": "archived"})
        self._audit(tenant_id, user_id, row.id, "archive", {"status": "archived"})
        return {"memory_id": row.id, "status": "archived"}

    def desensitize(self, tenant_id: str, user_id: str, memory_id: str) -> dict[str, Any]:
        row = self.resources.get("memory_items", tenant_id, memory_id)
        result = SecurityService.desensitize({"text": row.description})
        updated = self.resources.update("memory_items", tenant_id, user_id, memory_id, {"description": result["text"], "spec": {**(row.spec or {}), "desensitized": True}})
        self._audit(tenant_id, user_id, updated.id, "desensitize", {"text": result["text"]})
        return {"memory_id": updated.id, "status": "desensitized", "text": result["text"]}

    def cleanup_expired(self, tenant_id: str, user_id: str) -> dict[str, Any]:
        rows, _ = self.resources.list("memory_items", tenant_id, 1, 1000)
        now = datetime.now(timezone.utc)
        cleaned = []
        for row in rows:
            expires_at = (row.spec or {}).get("expires_at")
            if not expires_at:
                continue
            try:
                expires = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
            except ValueError:
                continue
            if expires <= now:
                self.resources.update("memory_items", tenant_id, user_id, row.id, {"status": "expired"})
                cleaned.append(row.id)
        job = self.resources.create("memory_cleanup_jobs", tenant_id, user_id, {"name": "memory cleanup", "status": "completed", "output_payload": {"memory_ids": cleaned}})
        return {"job_id": job.id, "cleaned": cleaned, "count": len(cleaned)}

    def resolve_conflict(self, tenant_id: str, user_id: str, conflict_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        conflict = self.resources.update(
            "memory_conflicts",
            tenant_id,
            user_id,
            conflict_id,
            {"status": "resolved", "output_payload": payload, "finished_at": datetime.now(timezone.utc)},
        )
        operation = self.resources.create(
            "memory_operations",
            tenant_id,
            user_id,
            {"name": "resolve conflict", "status": "completed", "parent_id": conflict.id, "output_payload": payload},
        )
        return {"conflict_id": conflict.id, "operation_id": operation.id, "status": "resolved"}

    def _audit(self, tenant_id: str, user_id: str, memory_id: str, action: str, payload: dict[str, Any]) -> None:
        self.resources.create(
            "memory_audit_logs",
            tenant_id,
            user_id,
            {"name": f"memory {action}", "status": "completed", "parent_id": memory_id, "resource_type": "memory_items", "resource_id": memory_id, "output_payload": payload},
        )

    def _find_by_hash(self, tenant_id: str, content_hash: str) -> Any | None:
        rows, _ = self.resources.list("memory_items", tenant_id, 1, 1000)
        for row in rows:
            if (row.spec or {}).get("hash") == content_hash and row.status == "active":
                return row
        return None

    @staticmethod
    def _visible(row: Any, spec: dict[str, Any], user_id: str, agent_id: str, session_id: str, team_id: str, include_shared: bool) -> bool:
        scope = str(spec.get("scope") or "")
        if scope == "session":
            return bool(session_id and (row.session_id == session_id or spec.get("session_id") == session_id))
        if scope == "user":
            return bool(row.user_id == user_id or spec.get("subject_user_id") == user_id)
        if scope == "agent":
            return bool(agent_id and (row.agent_id == agent_id or spec.get("agent_id") == agent_id))
        if scope == "team":
            return bool(team_id and (row.parent_id == team_id or spec.get("team_id") == team_id))
        if scope in {"tenant", "global", "shared"} or spec.get("shared"):
            return include_shared
        return include_shared if not scope else False

    @staticmethod
    def _hash(content: str, scope: str, agent_id: str, session_id: str, user_id: str, team_id: str) -> str:
        normalized = re.sub(r"\s+", " ", content.strip().lower())
        return hashlib.sha256(f"{scope}|{agent_id}|{session_id}|{user_id}|{team_id}|{normalized}".encode("utf-8")).hexdigest()

    @staticmethod
    def _score(query: str, text: str) -> float:
        if not query:
            return 1.0
        query_lower = query.lower()
        text_lower = text.lower()
        if query_lower in text_lower:
            return 1.0
        query_terms = {part for part in re.split(r"\W+", query_lower) if part}
        text_terms = {part for part in re.split(r"\W+", text_lower) if part}
        if not query_terms:
            return 0.0
        return round(len(query_terms & text_terms) / len(query_terms), 6)
