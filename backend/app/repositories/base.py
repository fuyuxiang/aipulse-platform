from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.db.mixins import utcnow


class SQLAlchemyRepository:
    def __init__(self, db: Session, model: type[Any]):
        self.db = db
        self.model = model

    def get(self, item_id: str, tenant_id: str | None = None, *, include_deleted: bool = False) -> Any:
        stmt = select(self.model).where(self.model.id == item_id)
        if tenant_id and hasattr(self.model, "tenant_id"):
            stmt = stmt.where(self.model.tenant_id == tenant_id)
        if hasattr(self.model, "deleted_at") and not include_deleted:
            stmt = stmt.where(self.model.deleted_at.is_(None))
        item = self.db.scalar(stmt)
        if item is None:
            raise AppError(ErrorCode.NOT_FOUND, f"{self.model.__tablename__} not found", 404)
        return item

    def list(
        self,
        tenant_id: str | None,
        *,
        page: int = 1,
        page_size: int = 20,
        filters: dict[str, Any] | None = None,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> tuple[list[Any], int]:
        stmt = select(self.model)
        count_stmt = select(func.count()).select_from(self.model)
        if tenant_id and hasattr(self.model, "tenant_id"):
            stmt = stmt.where(self.model.tenant_id == tenant_id)
            count_stmt = count_stmt.where(self.model.tenant_id == tenant_id)
        if hasattr(self.model, "deleted_at"):
            stmt = stmt.where(self.model.deleted_at.is_(None))
            count_stmt = count_stmt.where(self.model.deleted_at.is_(None))
        for key, value in (filters or {}).items():
            if value in (None, "") or not hasattr(self.model, key):
                continue
            column = getattr(self.model, key)
            if isinstance(value, str) and key in {"name", "code", "username"}:
                stmt = stmt.where(column.like(f"%{value}%"))
                count_stmt = count_stmt.where(column.like(f"%{value}%"))
            else:
                stmt = stmt.where(column == value)
                count_stmt = count_stmt.where(column == value)
        column = getattr(self.model, order_by, getattr(self.model, "created_at", self.model.id))
        stmt = stmt.order_by(column.desc() if descending else column.asc()).offset((page - 1) * page_size).limit(page_size)
        return list(self.db.scalars(stmt).all()), int(self.db.scalar(count_stmt) or 0)

    def create(self, values: dict[str, Any]) -> Any:
        item = self.model(**values)
        self.db.add(item)
        self.db.flush()
        return item

    def update(self, item: Any, values: dict[str, Any]) -> Any:
        for key, value in values.items():
            if hasattr(item, key):
                setattr(item, key, value)
        if hasattr(item, "updated_at"):
            item.updated_at = utcnow()
        self.db.flush()
        return item

    def soft_delete(self, item: Any, user_id: str) -> Any:
        if hasattr(item, "deleted_at"):
            item.deleted_at = utcnow()
            item.deleted_by = user_id
            if hasattr(item, "updated_by"):
                item.updated_by = user_id
            self.db.flush()
            return item
        self.db.delete(item)
        self.db.flush()
        return item

