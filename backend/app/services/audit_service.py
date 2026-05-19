from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
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

    def export(self, tenant_id: str, user_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        filters = filters or {}
        stmt = select(AuditLog).where(AuditLog.tenant_id == tenant_id).order_by(AuditLog.created_at.asc())
        if filters.get("action"):
            stmt = stmt.where(AuditLog.action == str(filters["action"]))
        if filters.get("resource_type"):
            stmt = stmt.where(AuditLog.resource_type == str(filters["resource_type"]))
        rows = list(self.db.scalars(stmt).all())
        exports_dir = settings.resolve_path(settings.data_dir) / "exports" / tenant_id
        exports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"audit-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.jsonl"
        path = exports_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = {column.name: getattr(row, column.name) for column in row.__table__.columns}
                handle.write(json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        from app.services.resource_service import ResourceService

        record = ResourceService(self.db).create(
            "audit_exports",
            tenant_id,
            user_id,
            {
                "name": filename,
                "status": "completed",
                "spec": {"path": str(path), "sha256": digest, "rows": len(rows), "filters": filters},
                "output_payload": {"path": str(path), "sha256": digest, "rows": len(rows)},
            },
        )
        return {"export_id": record.id, "path": str(path), "sha256": digest, "rows": len(rows)}
