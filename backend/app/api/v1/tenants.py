from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_permission
from app.core.response import ListResponse
from app.models.core import Tenant, User
from app.repositories.base import SQLAlchemyRepository
from app.schemas.auth import TenantCreate, TenantUpdate
from app.services.observe.audit_service import AuditService

router = APIRouter(prefix="/tenants", tags=["tenants"])


def _tenant(row: Tenant) -> dict[str, object]:
    return {
        "id": row.id,
        "code": row.code,
        "name": row.name,
        "status": row.status,
        "quota": row.quota,
        "settings": row.settings,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


@router.post("")
def create_tenant(payload: TenantCreate, db: Session = Depends(get_db), user: User = Depends(require_permission("tenants:write"))) -> dict[str, object]:
    row = SQLAlchemyRepository(db, Tenant).create({**payload.model_dump(), "created_by": user.id, "updated_by": user.id})
    AuditService(db).record(tenant_id=row.id, user_id=user.id, action="create", resource_type="tenants", resource_id=row.id, after_data=_tenant(row))
    db.commit()
    return _tenant(row)


@router.get("", response_model=ListResponse[dict[str, object]])
def list_tenants(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("tenants:read")),
) -> ListResponse[dict[str, object]]:
    rows, total = SQLAlchemyRepository(db, Tenant).list(None, page=page, page_size=page_size)
    return ListResponse(items=[_tenant(row) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/{tenant_id}")
def get_tenant(tenant_id: str, db: Session = Depends(get_db), _: User = Depends(require_permission("tenants:read"))) -> dict[str, object]:
    return _tenant(SQLAlchemyRepository(db, Tenant).get(tenant_id))


@router.put("/{tenant_id}")
def update_tenant(tenant_id: str, payload: TenantUpdate, db: Session = Depends(get_db), user: User = Depends(require_permission("tenants:write"))) -> dict[str, object]:
    repo = SQLAlchemyRepository(db, Tenant)
    row = repo.get(tenant_id)
    before = _tenant(row)
    row = repo.update(row, {**payload.model_dump(exclude_unset=True), "updated_by": user.id})
    AuditService(db).record(tenant_id=row.id, user_id=user.id, action="update", resource_type="tenants", resource_id=row.id, before_data=before, after_data=_tenant(row))
    db.commit()
    return _tenant(row)


@router.post("/{tenant_id}/disable")
def disable_tenant(tenant_id: str, db: Session = Depends(get_db), user: User = Depends(require_permission("tenants:write"))) -> dict[str, object]:
    return update_tenant(tenant_id, TenantUpdate(status="disabled"), db, user)


@router.delete("/{tenant_id}")
def delete_tenant(tenant_id: str, db: Session = Depends(get_db), user: User = Depends(require_permission("tenants:write"))) -> dict[str, str]:
    repo = SQLAlchemyRepository(db, Tenant)
    row = repo.get(tenant_id)
    repo.soft_delete(row, user.id)
    AuditService(db).record(tenant_id=tenant_id, user_id=user.id, action="delete", resource_type="tenants", resource_id=tenant_id)
    db.commit()
    return {"id": tenant_id, "status": "deleted"}
