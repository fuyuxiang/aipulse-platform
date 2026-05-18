from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.repositories.resources import ResourceRepository
from app.schemas.common import ActionResponse, ResourceCreate, ResourceUpdate
from app.services.audit_service import AuditService


class ResourceService:
    def __init__(self, db: Session):
        self.db = db
        self.audit = AuditService(db)

    def list(self, table: str, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[Any], int]:
        return ResourceRepository.for_table(self.db, table).list(tenant_id, page=page, page_size=page_size, filters=filters)

    def get(self, table: str, tenant_id: str, item_id: str) -> Any:
        return ResourceRepository.for_table(self.db, table).get(item_id, tenant_id)

    def create(self, table: str, tenant_id: str, user_id: str, payload: ResourceCreate | dict[str, Any]) -> Any:
        values = payload.model_dump() if isinstance(payload, ResourceCreate) else dict(payload)
        metadata = values.pop("metadata", None)
        if metadata is not None:
            values["metadata_json"] = metadata
        values["tenant_id"] = tenant_id
        values["created_by"] = user_id
        values["updated_by"] = user_id
        repo = ResourceRepository.for_table(self.db, table)
        values = {key: value for key, value in values.items() if hasattr(repo.model, key)}
        row = repo.create(values)
        self.audit.record(tenant_id=tenant_id, user_id=user_id, action="create", resource_type=table, resource_id=row.id, after_data=self.to_dict(row))
        self.db.commit()
        return row

    def update(self, table: str, tenant_id: str, user_id: str, item_id: str, payload: ResourceUpdate | dict[str, Any]) -> Any:
        repo = ResourceRepository.for_table(self.db, table)
        row = repo.get(item_id, tenant_id)
        before = self.to_dict(row)
        values = payload.model_dump(exclude_unset=True) if isinstance(payload, ResourceUpdate) else dict(payload)
        metadata = values.pop("metadata", None)
        if metadata is not None:
            values["metadata_json"] = metadata
        values["updated_by"] = user_id
        values = {key: value for key, value in values.items() if hasattr(repo.model, key)}
        row = repo.update(row, values)
        self.audit.record(tenant_id=tenant_id, user_id=user_id, action="update", resource_type=table, resource_id=row.id, before_data=before, after_data=self.to_dict(row))
        self.db.commit()
        return row

    def delete(self, table: str, tenant_id: str, user_id: str, item_id: str) -> dict[str, str]:
        repo = ResourceRepository.for_table(self.db, table)
        row = repo.get(item_id, tenant_id)
        before = self.to_dict(row)
        repo.soft_delete(row, user_id)
        self.audit.record(tenant_id=tenant_id, user_id=user_id, action="delete", resource_type=table, resource_id=item_id, before_data=before)
        self.db.commit()
        return {"id": item_id, "status": "deleted"}

    def action(
        self,
        table: str,
        tenant_id: str,
        user_id: str,
        *,
        action: str,
        resource_id: str = "",
        payload: dict[str, Any] | None = None,
        output_table: str | None = None,
    ) -> ActionResponse:
        payload = payload or {}
        target_id = resource_id
        if resource_id:
            self.get(table, tenant_id, resource_id)
        output: dict[str, Any] = {"payload": payload, "at": datetime.now(timezone.utc).isoformat()}
        if action in {"enable", "disable"} and resource_id:
            self.update(table, tenant_id, user_id, resource_id, {"enabled": action == "enable", "status": "active" if action == "enable" else "disabled"})
            output["enabled"] = action == "enable"
        elif action == "archive" and resource_id:
            self.update(table, tenant_id, user_id, resource_id, {"status": "archived"})
        elif action == "desensitize" and resource_id:
            row = self.get(table, tenant_id, resource_id)
            spec = dict(getattr(row, "spec", {}) or {})
            for key in list(spec):
                if any(token in key.lower() for token in ("secret", "password", "token", "key")):
                    spec[key] = "***"
            self.update(table, tenant_id, user_id, resource_id, {"spec": spec})
        elif output_table:
            created = self.create(
                output_table,
                tenant_id,
                user_id,
                {
                    "name": payload.get("name", action),
                    "code": payload.get("code", ""),
                    "status": "completed",
                    "resource_type": table,
                    "parent_id": resource_id,
                    "agent_id": payload.get("agent_id", resource_id if "agent" in table else ""),
                    "workflow_id": payload.get("workflow_id", resource_id if "workflow" in table else ""),
                    "model_id": payload.get("model_id", resource_id if table == "models" else ""),
                    "config": payload.get("config", {}),
                    "spec": payload,
                    "input_payload": payload,
                    "output_payload": {"action": action, "status": "completed"},
                },
            )
            target_id = created.id
            output["created_id"] = created.id
        else:
            self.audit.record(tenant_id=tenant_id, user_id=user_id, action=action, resource_type=table, resource_id=resource_id, after_data=payload)
            self.db.commit()
        return ActionResponse(id=target_id or "", action=action, status="completed", resource_type=table, resource_id=resource_id, output=output)

    @staticmethod
    def to_dict(row: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for column in row.__table__.columns:
            value = getattr(row, column.name)
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            data[column.name] = value
        return data

    def require_embedding_dimensions(self, kb: Any, embedding: list[float]) -> None:
        expected = int((kb.config or {}).get("embedding_dimensions") or 0)
        if expected and len(embedding) != expected:
            raise AppError(ErrorCode.VALIDATION_ERROR, f"embedding dimension mismatch: expected {expected}, got {len(embedding)}", 422)
