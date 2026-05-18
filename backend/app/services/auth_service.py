from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.constants import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USERNAME, DEFAULT_TENANT_CODE, ErrorCode
from app.core.errors import AppError
from app.core.security import create_token, decode_token, hash_password, hash_secret, verify_password
from app.db.mixins import utcnow
from app.models.core import APIKey, LoginLog, Permission, RevokedToken, Role, RolePermission, Tenant, User, UserRole
from app.services.audit_service import AuditService


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.audit = AuditService(db)

    def authenticate(self, tenant_code: str, username: str, password: str, ip: str = "", user_agent: str = "") -> tuple[User, Tenant]:
        tenant = self.db.scalar(select(Tenant).where(Tenant.code == tenant_code, Tenant.deleted_at.is_(None)))
        if tenant is None:
            raise AppError(ErrorCode.UNAUTHORIZED, "invalid tenant or credentials", 401)
        if tenant.status != "active":
            raise AppError(ErrorCode.TENANT_INACTIVE, "tenant is inactive", 403)
        user = self.db.scalar(select(User).where(User.tenant_id == tenant.id, User.username == username, User.deleted_at.is_(None)))
        ok = bool(user and user.is_active and verify_password(password, user.password_hash))
        self.db.add(
            LoginLog(
                tenant_id=tenant.id,
                username=username,
                user_id=user.id if user else "",
                status="success" if ok else "failed",
                ip_address=ip,
                user_agent=user_agent,
                failure_reason="" if ok else "invalid credentials",
            )
        )
        if not ok:
            if user:
                user.failed_login_count += 1
            self.db.commit()
            raise AppError(ErrorCode.UNAUTHORIZED, "invalid tenant or credentials", 401)
        assert user is not None
        user.failed_login_count = 0
        user.last_login_at = utcnow()
        self.audit.record(tenant_id=tenant.id, user_id=user.id, action="login", resource_type="users", resource_id=user.id, ip_address=ip, user_agent=user_agent)
        self.db.commit()
        return user, tenant

    def issue_tokens(self, user: User, tenant: Tenant) -> dict[str, Any]:
        access_seconds = settings.access_token_minutes * 60
        refresh_seconds = settings.refresh_token_days * 24 * 3600
        return {
            "access_token": create_token(user.id, tenant.id, "access", access_seconds, {"username": user.username}),
            "refresh_token": create_token(user.id, tenant.id, "refresh", refresh_seconds, {"username": user.username}),
            "expires_in": access_seconds,
            "must_change_password": user.must_change_password,
        }

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        payload = decode_token(refresh_token, "refresh")
        self._ensure_not_revoked(payload["jti"], payload["tenant_id"])
        user = self.db.get(User, payload["sub"])
        tenant = self.db.get(Tenant, payload["tenant_id"])
        if user is None or tenant is None:
            raise AppError(ErrorCode.UNAUTHORIZED, "token principal not found", 401)
        self.audit.record(tenant_id=tenant.id, user_id=user.id, action="refresh_token", resource_type="users", resource_id=user.id)
        self.db.commit()
        return self.issue_tokens(user, tenant)

    def revoke(self, token: str, current_user: User | None = None) -> dict[str, str]:
        payload = decode_token(token)
        if not self.db.scalar(select(RevokedToken).where(RevokedToken.token_jti == payload["jti"], RevokedToken.tenant_id == payload["tenant_id"])):
            self.db.add(
                RevokedToken(
                    tenant_id=payload["tenant_id"],
                    token_jti=payload["jti"],
                    token_type=payload.get("type", ""),
                    expires_at=utcnow() + timedelta(seconds=max(0, int(payload["exp"]) - int(payload["iat"]))),
                    created_by=current_user.id if current_user else payload["sub"],
                    updated_by=current_user.id if current_user else payload["sub"],
                )
            )
        self.audit.record(tenant_id=payload["tenant_id"], user_id=current_user.id if current_user else payload["sub"], action="revoke_token", resource_type="revoked_tokens")
        self.db.commit()
        return {"status": "revoked"}

    def authenticate_token(self, token: str) -> User:
        payload = decode_token(token, "access")
        self._ensure_not_revoked(payload["jti"], payload["tenant_id"])
        user = self.db.get(User, payload["sub"])
        if user is None or not user.is_active or user.deleted_at is not None:
            raise AppError(ErrorCode.UNAUTHORIZED, "inactive user", 401)
        return user

    def authenticate_api_key(self, api_key: str) -> User:
        key_hash = hash_secret(api_key)
        row = self.db.scalar(select(APIKey).where(APIKey.key_hash == key_hash, APIKey.enabled.is_(True), APIKey.deleted_at.is_(None)))
        if row is None:
            raise AppError(ErrorCode.UNAUTHORIZED, "invalid api key", 401)
        user = self.db.get(User, row.user_id)
        if user is None:
            raise AppError(ErrorCode.UNAUTHORIZED, "api key owner not found", 401)
        row.last_used_at = utcnow()
        self.db.commit()
        return user

    def create_api_key(self, tenant_id: str, user_id: str, name: str, scopes: list[str]) -> dict[str, str]:
        raw = f"ak_{secrets.token_urlsafe(32)}"
        row = APIKey(tenant_id=tenant_id, user_id=user_id, name=name, key_hash=hash_secret(raw), scopes=scopes, created_by=user_id, updated_by=user_id)
        self.db.add(row)
        self.audit.record(tenant_id=tenant_id, user_id=user_id, action="create_api_key", resource_type="api_keys", resource_id=row.id)
        self.db.commit()
        return {"id": row.id, "api_key": raw}

    def permissions_for(self, user: User) -> list[str]:
        if user.is_superuser:
            return ["*"]
        stmt = (
            select(Permission.code)
            .join(RolePermission, RolePermission.permission_id == Permission.id)
            .join(UserRole, UserRole.role_id == RolePermission.role_id)
            .where(UserRole.user_id == user.id, Permission.tenant_id == user.tenant_id)
        )
        return list(self.db.scalars(stmt).all())

    def _ensure_not_revoked(self, jti: str, tenant_id: str) -> None:
        revoked = self.db.scalar(select(RevokedToken).where(RevokedToken.tenant_id == tenant_id, RevokedToken.token_jti == jti))
        if revoked is not None:
            raise AppError(ErrorCode.UNAUTHORIZED, "token revoked", 401)


def ensure_default_identity(db: Session) -> dict[str, str]:
    tenant = db.scalar(select(Tenant).where(Tenant.code == DEFAULT_TENANT_CODE))
    if tenant is None:
        tenant = Tenant(code=DEFAULT_TENANT_CODE, name="Default Tenant", status="active", quota={"agents": 100, "models": 100})
        db.add(tenant)
        db.flush()
    user = db.scalar(select(User).where(User.tenant_id == tenant.id, User.username == DEFAULT_ADMIN_USERNAME))
    if user is None:
        user = User(
            tenant_id=tenant.id,
            username=DEFAULT_ADMIN_USERNAME,
            display_name="Platform Admin",
            email="admin@local",
            password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            is_active=True,
            is_superuser=True,
            must_change_password=True,
            created_by="system",
            updated_by="system",
        )
        db.add(user)
        db.flush()
    role = db.scalar(select(Role).where(Role.tenant_id == tenant.id, Role.code == "admin"))
    if role is None:
        role = Role(tenant_id=tenant.id, code="admin", name="Administrator", description="Full platform administration", created_by=user.id, updated_by=user.id)
        db.add(role)
        db.flush()
    if db.scalar(select(UserRole).where(UserRole.tenant_id == tenant.id, UserRole.user_id == user.id, UserRole.role_id == role.id)) is None:
        db.add(UserRole(tenant_id=tenant.id, user_id=user.id, role_id=role.id, created_by=user.id, updated_by=user.id))
    db.commit()
    return {"tenant_id": tenant.id, "user_id": user.id}
