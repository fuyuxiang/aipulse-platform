from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService
from app.services.security_service import SecurityService


class MemoryService:
    def __init__(self, db: Session):
        self.resources = ResourceService(db)

    def search(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or payload.get("text") or "").lower()
        scope = str(payload.get("scope") or "")
        agent_id = str(payload.get("agent_id") or "")
        session_id = str(payload.get("session_id") or "")
        rows, _ = self.resources.list("memory_items", tenant_id, 1, int(payload.get("limit") or 200))
        matches = []
        for row in rows:
            spec = dict(row.spec or {})
            if scope and spec.get("scope") != scope:
                continue
            if agent_id and row.agent_id != agent_id and spec.get("agent_id") != agent_id:
                continue
            if session_id and row.session_id != session_id and spec.get("session_id") != session_id:
                continue
            haystack = f"{row.name} {row.description} {row.spec}".lower()
            if not query or query in haystack:
                matches.append(ResourceService.to_dict(row))
        operation = self.resources.create(
            "memory_operations",
            tenant_id,
            user_id,
            {"name": "memory search", "status": "completed", "input_payload": payload, "output_payload": {"count": len(matches)}},
        )
        return {"operation_id": operation.id, "total": len(matches), "items": matches}

    def extract(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or "")
        scope = str(payload.get("scope") or "session")
        form = str(payload.get("form") or "episodic")
        facts = [part.strip(" .") for part in re.split(r"[。\n.;]+", text) if len(part.strip()) > 2]
        created = []
        for fact in facts:
            row = self.resources.create(
                "memory_items",
                tenant_id,
                user_id,
                {
                    "name": fact[:120],
                    "description": fact,
                    "status": "active",
                    "agent_id": str(payload.get("agent_id") or ""),
                    "session_id": str(payload.get("session_id") or ""),
                    "user_id": str(payload.get("subject_user_id") or user_id),
                    "spec": {"scope": scope, "form": form, "source": "extraction", "confidence": float(payload.get("confidence") or 0.8)},
                },
            )
            created.append(row.id)
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
