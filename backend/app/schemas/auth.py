from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    tenant: str = "default"
    username: str
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    must_change_password: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


class RevokeTokenRequest(BaseModel):
    token: str


class TenantCreate(BaseModel):
    code: str
    name: str
    quota: dict[str, object] = Field(default_factory=dict)
    settings: dict[str, object] = Field(default_factory=dict)


class TenantUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    quota: dict[str, object] | None = None
    settings: dict[str, object] | None = None


class UserCreate(BaseModel):
    username: str
    password: str
    display_name: str = ""
    email: str = ""
    is_superuser: bool = False


class UserUpdate(BaseModel):
    display_name: str | None = None
    email: str | None = None
    is_active: bool | None = None
    is_superuser: bool | None = None


class CurrentUser(BaseModel):
    id: str
    tenant_id: str
    username: str
    display_name: str
    email: str
    is_superuser: bool
    must_change_password: bool
    permissions: list[str]
    login_at: datetime | None = None

