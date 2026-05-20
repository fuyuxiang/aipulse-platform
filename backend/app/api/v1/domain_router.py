from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_tenant_id, require_permission
from app.core.response import ListResponse
from app.models.core import User
from app.schemas.common import ActionRequest, ActionResponse, ResourceCreate, ResourceRead, ResourceUpdate
from app.services._shared.resource_service import ResourceService


def _read(row: Any) -> ResourceRead:
    data = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    return ResourceRead.model_validate(data)


def add_crud_routes(router: APIRouter, *, table: str, prefix: str, permission: str) -> None:
    @router.post(prefix, response_model=ResourceRead)
    def create_resource(
        payload: ResourceCreate,
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        user: User = Depends(require_permission(f"{permission}:write")),
    ) -> ResourceRead:
        return _read(ResourceService(db).create(table, tenant_id, user.id, payload))

    @router.get(prefix, response_model=ListResponse[ResourceRead])
    def list_resources(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
        name: str = "",
        status: str = "",
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        _: User = Depends(require_permission(f"{permission}:read")),
    ) -> ListResponse[ResourceRead]:
        rows, total = ResourceService(db).list(table, tenant_id, page, page_size, {"name": name, "status": status})
        return ListResponse(items=[_read(row) for row in rows], total=total, page=page, page_size=page_size)

    @router.get(f"{prefix}/{{item_id}}", response_model=ResourceRead)
    def get_resource(
        item_id: str,
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        _: User = Depends(require_permission(f"{permission}:read")),
    ) -> ResourceRead:
        return _read(ResourceService(db).get(table, tenant_id, item_id))

    @router.put(f"{prefix}/{{item_id}}", response_model=ResourceRead)
    def update_resource(
        item_id: str,
        payload: ResourceUpdate,
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        user: User = Depends(require_permission(f"{permission}:write")),
    ) -> ResourceRead:
        return _read(ResourceService(db).update(table, tenant_id, user.id, item_id, payload))

    @router.delete(f"{prefix}/{{item_id}}")
    def delete_resource(
        item_id: str,
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        user: User = Depends(require_permission(f"{permission}:write")),
    ) -> dict[str, str]:
        return ResourceService(db).delete(table, tenant_id, user.id, item_id)


def add_action_route(
    router: APIRouter,
    *,
    method: str,
    path: str,
    table: str,
    permission: str,
    action: str,
    output_table: str | None = None,
) -> None:
    async def endpoint(
        request: Request,
        payload: ActionRequest = Body(default_factory=ActionRequest),
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        user: User = Depends(require_permission(f"{permission}:write")),
    ) -> ActionResponse:
        resource_id = next((value for key, value in request.path_params.items() if key.endswith("_id") or key == "item_id"), "")
        effective_action = payload.action or action
        return ResourceService(db).action(
            table,
            tenant_id,
            user.id,
            action=effective_action,
            resource_id=resource_id,
            payload=payload.payload,
            output_table=output_table,
        )

    router.add_api_route(path, endpoint, methods=[method.upper()], response_model=ActionResponse)


def add_list_route(
    router: APIRouter,
    *,
    method: str,
    path: str,
    table: str,
    permission: str,
    filters_from_path: Callable[[Request], dict[str, Any]] | None = None,
) -> None:
    async def endpoint(
        request: Request,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=200),
        db: Session = Depends(get_db),
        tenant_id: str = Depends(get_tenant_id),
        _: User = Depends(require_permission(f"{permission}:read")),
    ) -> ListResponse[ResourceRead]:
        filters = filters_from_path(request) if filters_from_path else {}
        rows, total = ResourceService(db).list(table, tenant_id, page, page_size, filters)
        return ListResponse(items=[_read(row) for row in rows], total=total, page=page, page_size=page_size)

    router.add_api_route(path, endpoint, methods=[method.upper()], response_model=ListResponse[ResourceRead])


def public_ping_router() -> APIRouter:
    router = APIRouter()

    @router.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    return router
