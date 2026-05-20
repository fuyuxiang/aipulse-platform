from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.repositories.resources import ResourceRepository
from app.schemas.common import ActionResponse, ResourceCreate, ResourceUpdate
from app.services.observe.audit_service import AuditService


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
        values = self._prepare_sensitive_values(table, values)
        values["tenant_id"] = tenant_id
        values["created_by"] = user_id
        values["updated_by"] = user_id
        repo = ResourceRepository.for_table(self.db, table)
        values = {key: value for key, value in values.items() if hasattr(repo.model, key)}
        row = repo.create(values)
        self.audit.record(tenant_id=tenant_id, user_id=user_id, action="create", resource_type=table, resource_id=row.id, after_data=self.to_dict(row))
        self._evaluate_alert_rules(table, tenant_id, user_id, row)
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
        values = self._prepare_sensitive_values(table, values)
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

    def require_embedding_dimensions(self, kb: Any, embedding: Sequence[float]) -> None:
        expected = int((kb.config or {}).get("embedding_dimensions") or 0)
        if expected and len(embedding) != expected:
            raise AppError(ErrorCode.VALIDATION_ERROR, f"embedding dimension mismatch: expected {expected}, got {len(embedding)}", 422)

    @staticmethod
    def _prepare_sensitive_values(table: str, values: dict[str, Any]) -> dict[str, Any]:
        if table not in {"model_credentials", "secret_refs"}:
            return values
        prepared = dict(values)
        for field in ("config", "spec", "metadata_json", "input_payload"):
            if isinstance(prepared.get(field), dict):
                prepared[field] = ResourceService._redact_secret_dict(dict(prepared[field]))
        secret_value = str(values.get("secret_value") or values.get("value") or "")
        for field in ("config", "spec"):
            source = values.get(field)
            if isinstance(source, dict):
                secret_value = secret_value or str(source.get("secret_value") or source.get("value") or source.get("api_key") or "")
        if secret_value:
            digest = hmac.new(settings.jwt_secret.encode("utf-8"), secret_value.encode("utf-8"), hashlib.sha256).hexdigest()
            spec = dict(prepared.get("spec") or {})
            spec.update({"secret_sha256": digest, "secret_ref": spec.get("secret_ref") or f"local://{table}/{digest[:16]}", "has_secret": True})
            for key in ("secret_value", "value", "api_key", "token", "password"):
                spec.pop(key, None)
            prepared["spec"] = spec
        return prepared

    @staticmethod
    def _redact_secret_dict(payload: dict[str, Any]) -> dict[str, Any]:
        for key in list(payload):
            if any(token in key.lower() for token in ("secret", "password", "token", "api_key", "apikey")):
                value = str(payload.get(key) or "")
                if key.endswith("_ref") or key.endswith("_id"):
                    continue
                payload[key] = "***" if value else ""
        return payload

    def _evaluate_alert_rules(self, table: str, tenant_id: str, user_id: str, row: Any) -> None:
        if table == "alert_events" or table not in {"runtime_metrics", "model_call_logs", "tool_call_logs", "knowledge_retrieval_logs", "workflow_runs", "agent_run_records"}:
            return
        rules, _ = self.list("alert_rules", tenant_id, 1, 200, {"status": "active"})
        for rule in rules:
            config = {**(rule.config or {}), **(rule.spec or {})}
            source_table = str(config.get("source_table") or "")
            if source_table and source_table != table:
                continue
            field = str(config.get("field") or "latency_ms")
            actual = self._metric_value(row, field)
            if actual is None:
                continue
            threshold = float(config.get("threshold") or 0)
            operator = str(config.get("operator") or "gt")
            if self._compare_metric(actual, threshold, operator):
                self.create(
                    "alert_events",
                    tenant_id,
                    user_id,
                    {
                        "name": rule.name,
                        "status": "triggered",
                        "parent_id": rule.id,
                        "resource_type": table,
                        "spec": {"rule_id": rule.id, "field": field, "operator": operator, "threshold": threshold, "actual": actual},
                        "input_payload": self.to_dict(row),
                    },
                )

    @staticmethod
    def _metric_value(row: Any, field: str) -> float | None:
        if hasattr(row, field):
            value = getattr(row, field)
        else:
            value = (row.spec or {}).get(field) or (row.output_payload or {}).get(field) or (row.input_payload or {}).get(field)
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _compare_metric(actual: float, threshold: float, operator: str) -> bool:
        if operator == "gte":
            return actual >= threshold
        if operator == "lt":
            return actual < threshold
        if operator == "lte":
            return actual <= threshold
        if operator == "eq":
            return actual == threshold
        return actual > threshold
