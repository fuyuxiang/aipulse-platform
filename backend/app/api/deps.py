from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.core.tracing import tenant_id_var, user_id_var
from app.db.session import get_db
from app.models.core import User
from app.services.auth_service import AuthService
from app.services.rbac_service import RBACService

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    request: Request,
    db: DbSession,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> User:
    auth = AuthService(db)
    if api_key:
        user = auth.authenticate_api_key(api_key)
    else:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AppError(ErrorCode.UNAUTHORIZED, "missing bearer token", 401)
        user = auth.authenticate_token(authorization.split(" ", 1)[1].strip())
    request.state.user = user
    request.state.tenant_id = user.tenant_id
    tenant_id_var.set(user.tenant_id)
    user_id_var.set(user.id)
    return user


CurrentUserDep = Annotated[User, Depends(get_current_user)]


def get_tenant_id(current_user: CurrentUserDep, x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None) -> str:
    if x_tenant_id and x_tenant_id != current_user.tenant_id and not current_user.is_superuser:
        raise AppError(ErrorCode.FORBIDDEN, "cross-tenant access requires platform administrator", 403)
    tenant_id = x_tenant_id or current_user.tenant_id
    tenant_id_var.set(tenant_id)
    return tenant_id


TenantIdDep = Annotated[str, Depends(get_tenant_id)]


def require_permission(permission: str) -> Callable[[DbSession, CurrentUserDep], User]:
    def dependency(db: DbSession, current_user: CurrentUserDep) -> User:
        RBACService(AuthService(db)).require(current_user, permission)
        return current_user

    return dependency

