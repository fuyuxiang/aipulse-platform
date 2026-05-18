from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.models.core import APIKey, AuditLog, LoginLog, Organization, OrganizationMember, Permission, RevokedToken, Role, RolePermission, ServiceAccount, SystemConfig, UserRole
from app.models.resources import RESOURCE_MODELS
from app.repositories.base import SQLAlchemyRepository

CORE_RESOURCE_MODELS = {
    "organizations": Organization,
    "organization_members": OrganizationMember,
    "roles": Role,
    "permissions": Permission,
    "role_permissions": RolePermission,
    "user_roles": UserRole,
    "system_configs": SystemConfig,
    "audit_logs": AuditLog,
    "api_keys": APIKey,
    "service_accounts": ServiceAccount,
    "revoked_tokens": RevokedToken,
    "login_logs": LoginLog,
}


class ResourceRepository(SQLAlchemyRepository):
    @classmethod
    def for_table(cls, db: Session, table_name: str) -> "ResourceRepository":
        model = RESOURCE_MODELS.get(table_name) or CORE_RESOURCE_MODELS.get(table_name)
        if model is None:
            raise AppError(ErrorCode.NOT_FOUND, f"resource table not registered: {table_name}", 404)
        return cls(db, model)
