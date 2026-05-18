from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import IdMixin, SoftDeleteMixin, TenantScopedMixin, TimestampMixin, json_default, utcnow


class Tenant(Base, IdMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "tenants"
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True, nullable=False)
    quota: Mapped[dict[str, Any]] = mapped_column(JSON, default=json_default, nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=json_default, nullable=False)


class User(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "username", name="uq_users_tenant_username"),)
    username: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    email: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Organization(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "organizations"
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)


class OrganizationMember(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "organization_members"
    organization_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(128), default="", nullable=False)


class Role(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "roles"
    __table_args__ = (UniqueConstraint("tenant_id", "code", name="uq_roles_tenant_code"),)
    code: Mapped[str] = mapped_column(String(96), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    data_scope: Mapped[str] = mapped_column(String(64), default="tenant", nullable=False)


class Permission(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "permissions"
    __table_args__ = (UniqueConstraint("tenant_id", "code", name="uq_permissions_tenant_code"),)
    code: Mapped[str] = mapped_column(String(160), nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    permission_type: Mapped[str] = mapped_column(String(64), default="api", nullable=False)
    resource_type: Mapped[str] = mapped_column(String(96), default="", nullable=False)
    action: Mapped[str] = mapped_column(String(96), default="", nullable=False)


class RolePermission(Base, IdMixin, TenantScopedMixin, TimestampMixin):
    __tablename__ = "role_permissions"
    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id"), index=True, nullable=False)
    permission_id: Mapped[str] = mapped_column(ForeignKey("permissions.id"), index=True, nullable=False)


class UserRole(Base, IdMixin, TenantScopedMixin, TimestampMixin):
    __tablename__ = "user_roles"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    role_id: Mapped[str] = mapped_column(ForeignKey("roles.id"), index=True, nullable=False)


class APIKey(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "api_keys"
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ServiceAccount(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "service_accounts"
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class RevokedToken(Base, IdMixin, TenantScopedMixin, TimestampMixin):
    __tablename__ = "revoked_tokens"
    token_jti: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    token_type: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LoginLog(Base, IdMixin, TenantScopedMixin, TimestampMixin):
    __tablename__ = "login_logs"
    username: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    ip_address: Mapped[str] = mapped_column(String(96), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, default="", nullable=False)
    failure_reason: Mapped[str] = mapped_column(Text, default="", nullable=False)


class AuditLog(Base, IdMixin, TenantScopedMixin, TimestampMixin):
    __tablename__ = "audit_logs"
    user_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    action: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    resource_type: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    resource_id: Mapped[str] = mapped_column(String(96), default="", index=True, nullable=False)
    before_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=json_default, nullable=False)
    after_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=json_default, nullable=False)
    ip_address: Mapped[str] = mapped_column(String(96), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, default="", nullable=False)
    trace_id: Mapped[str] = mapped_column(String(96), default="", index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="success", nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    hash: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(128), default="", nullable=False)


class SystemConfig(Base, IdMixin, TenantScopedMixin, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "system_configs"
    key: Mapped[str] = mapped_column(String(160), index=True, nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=json_default, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

