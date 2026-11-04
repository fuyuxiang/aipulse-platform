from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy import select
from app.models.core import UserRole
from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.services.settings.role_service import RoleService
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.core.response import ListResponse
from app.core.security import hash_password
from app.models.core import User
from app.repositories.base import SQLAlchemyRepository
from app.schemas.auth import UserCreate, UserUpdate
from app.services.observe.audit_service import AuditService

router = APIRouter(prefix="/users", tags=["users"])


def _user(row: User) -> dict[str, object]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "username": row.username,
        "display_name": row.display_name,
        "email": row.email,
        "is_active": row.is_active,
        "is_superuser": row.is_superuser,
        "must_change_password": row.must_change_password,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@router.post("")
def create_user(payload: UserCreate, tenant_id: TenantIdDep, db: Session = Depends(get_db), user: User = Depends(require_permission("users:write"))) -> dict[str, object]:
    if payload.is_superuser and not user.is_superuser:
        raise AppError(ErrorCode.FORBIDDEN, "only platform administrators can grant superuser", 403)
    values = payload.model_dump(exclude={"password"})
    values.update({"tenant_id": tenant_id, "password_hash": hash_password(payload.password), "created_by": user.id, "updated_by": user.id})
    row = SQLAlchemyRepository(db, User).create(values)
    AuditService(db).record(tenant_id=tenant_id, user_id=user.id, action="create", resource_type="users", resource_id=row.id, after_data=_user(row))
    db.commit()
    return _user(row)


@router.get("", response_model=ListResponse[dict[str, object]])
def list_users(tenant_id: TenantIdDep, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200), db: Session = Depends(get_db), _: User = Depends(require_permission("users:read"))) -> ListResponse[dict[str, object]]:
    rows, total = SQLAlchemyRepository(db, User).list(tenant_id, page=page, page_size=page_size)
    return ListResponse(items=[_user(row) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/{user_id}")
def get_user(user_id: str, tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("users:read"))) -> dict[str, object]:
    return _user(SQLAlchemyRepository(db, User).get(user_id, tenant_id))


@router.put("/{user_id}")
def update_user(user_id: str, payload: UserUpdate, tenant_id: TenantIdDep, db: Session = Depends(get_db), actor: User = Depends(require_permission("users:write"))) -> dict[str, object]:
    repo = SQLAlchemyRepository(db, User)
    row = repo.get(user_id, tenant_id)
    if not actor.is_superuser and (row.is_superuser or payload.is_superuser):
        raise AppError(ErrorCode.FORBIDDEN, "only platform administrators can manage superusers", 403)
    before = _user(row)
    row = repo.update(row, {**payload.model_dump(exclude_unset=True), "updated_by": actor.id})
    AuditService(db).record(tenant_id=tenant_id, user_id=actor.id, action="update", resource_type="users", resource_id=row.id, before_data=before, after_data=_user(row))
    db.commit()
    return _user(row)


@router.delete("/{user_id}")
def delete_user(user_id: str, tenant_id: TenantIdDep, db: Session = Depends(get_db), actor: User = Depends(require_permission("users:write"))) -> dict[str, str]:
    repo = SQLAlchemyRepository(db, User)
    row = repo.get(user_id, tenant_id)
    if row.is_superuser and not actor.is_superuser:
        raise AppError(ErrorCode.FORBIDDEN, "only platform administrators can manage superusers", 403)
    repo.soft_delete(row, actor.id)
    AuditService(db).record(tenant_id=tenant_id, user_id=actor.id, action="delete", resource_type="users", resource_id=user_id)
    db.commit()
    return {"id": user_id, "status": "deleted"}


@router.get("/{user_id}/roles")
def user_roles(user_id: str, tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("roles:read"))) -> dict[str, object]:
    SQLAlchemyRepository(db, User).get(user_id, tenant_id)
    return {"role_ids": list(db.scalars(select(UserRole.role_id).where(UserRole.tenant_id == tenant_id, UserRole.user_id == user_id)))}


@router.put("/{user_id}/roles")
def set_user_roles(user_id: str, tenant_id: TenantIdDep, role_ids: list[str] = Body(), db: Session = Depends(get_db), actor: User = Depends(require_permission("roles:write"))) -> dict[str, object]:
    RoleService(db).set_user_roles(tenant_id, user_id, role_ids)
    AuditService(db).record(tenant_id=tenant_id, user_id=actor.id, action="set_roles", resource_type="users", resource_id=user_id, after_data={"role_ids": role_ids})
    db.commit()
    return {"user_id": user_id, "role_ids": role_ids}
