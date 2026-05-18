from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.tracing import current_trace_id
from app.models.core import AuditLog


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


class AuditService:
    def __init__(self, db: Session):
        self.db = db

    def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str = "",
        before_data: dict[str, Any] | None = None,
        after_data: dict[str, Any] | None = None,
        status: str = "success",
        error_message: str = "",
        ip_address: str = "",
        user_agent: str = "",
    ) -> AuditLog:
        previous = self.db.scalar(select(AuditLog).where(AuditLog.tenant_id == tenant_id).order_by(AuditLog.created_at.desc()))
        previous_hash = previous.hash if previous else ""
        body = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "before_data": before_data or {},
            "after_data": after_data or {},
            "trace_id": current_trace_id(),
            "status": status,
            "error_message": error_message,
            "previous_hash": previous_hash,
        }
        digest = hashlib.sha256(_stable_json(body).encode("utf-8")).hexdigest()
        row = AuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            before_data=before_data or {},
            after_data=after_data or {},
            ip_address=ip_address,
            user_agent=user_agent,
            trace_id=current_trace_id(),
            status=status,
            error_message=error_message,
            previous_hash=previous_hash,
            hash=digest,
            created_by=user_id,
            updated_by=user_id,
        )
        self.db.add(row)
        self.db.flush()
        return row

    def verify_chain(self, tenant_id: str) -> dict[str, Any]:
        rows = list(self.db.scalars(select(AuditLog).where(AuditLog.tenant_id == tenant_id).order_by(AuditLog.created_at.asc())).all())
        previous_hash = ""
        broken: list[str] = []
        for row in rows:
            body = {
                "tenant_id": row.tenant_id,
                "user_id": row.user_id,
                "action": row.action,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "before_data": row.before_data,
                "after_data": row.after_data,
                "trace_id": row.trace_id,
                "status": row.status,
                "error_message": row.error_message,
                "previous_hash": row.previous_hash,
            }
            expected = hashlib.sha256(_stable_json(body).encode("utf-8")).hexdigest()
            if row.previous_hash != previous_hash or row.hash != expected:
                broken.append(row.id)
            previous_hash = row.hash
        return {"valid": not broken, "checked": len(rows), "broken_ids": broken}

