"""RBAC writes use the same relations that request authorization reads."""
from __future__ import annotations

from typing import Any
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.models.core import Permission, Role, RolePermission, User, UserRole
from app.repositories.base import SQLAlchemyRepository


class RoleService:
    def __init__(self, db: Session):
        self.db = db

    def validate_binding(self, table: str, tenant_id: str, values: dict[str, Any]) -> None:
        fields = {"user_roles": (("user_id", User), ("role_id", Role)),
                  "role_permissions": (("role_id", Role), ("permission_id", Permission))}
        for field, model in fields.get(table, ()):
            SQLAlchemyRepository(self.db, model).get(str(values.get(field) or ""), tenant_id)

    def set_permissions(self, tenant_id: str, role: Role, codes: list[str]) -> None:
        permissions = list(self.db.scalars(select(Permission).where(
            Permission.tenant_id == tenant_id, Permission.code.in_(codes), Permission.deleted_at.is_(None))))
        if set(codes) != {p.code for p in permissions}:
            raise AppError(ErrorCode.VALIDATION_ERROR, "unknown permission code", 422)
        self.db.execute(delete(RolePermission).where(RolePermission.tenant_id == tenant_id, RolePermission.role_id == role.id))
        for permission in permissions:
            self.db.add(RolePermission(tenant_id=tenant_id, role_id=role.id, permission_id=permission.id))

    def read(self, role: Role) -> dict[str, Any]:
        codes = list(self.db.scalars(select(Permission.code).join(RolePermission, RolePermission.permission_id == Permission.id).where(
            RolePermission.tenant_id == role.tenant_id, RolePermission.role_id == role.id, Permission.deleted_at.is_(None))))
        return {"permissions": codes, "scope": role.data_scope}

    def set_user_roles(self, tenant_id: str, user_id: str, role_ids: list[str]) -> None:
        SQLAlchemyRepository(self.db, User).get(user_id, tenant_id)
        for role_id in set(role_ids):
            SQLAlchemyRepository(self.db, Role).get(role_id, tenant_id)
        self.db.execute(delete(UserRole).where(UserRole.tenant_id == tenant_id, UserRole.user_id == user_id))
        for role_id in set(role_ids):
            self.db.add(UserRole(tenant_id=tenant_id, user_id=user_id, role_id=role_id))
