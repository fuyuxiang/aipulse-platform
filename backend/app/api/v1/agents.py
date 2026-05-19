from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.schemas.common import ResourceRead
from app.services.agent_service import AgentService
from app.services.agent_publication_service import AgentPublicationService
from app.services.resource_service import ResourceService

router = APIRouter(tags=["agents"])

add_crud_routes(router, table="agents", prefix="/agents", permission="agents")
add_crud_routes(router, table="agent_templates", prefix="/agent-templates", permission="agents")

@router.post("/agents/{agent_id}/clone")
def clone_agent(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentService(db).clone(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/{agent_id}/versions")
def create_agent_version(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentService(db).create_version(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/{agent_id}/release")
def release_agent(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentService(db).release(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/{agent_id}/gray-release")
def gray_release_agent(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentService(db).gray_release(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/{agent_id}/rollback")
def rollback_agent(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentService(db).rollback(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/{agent_id}/debug-run")
async def debug_run_agent(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await AgentService(db).debug_run(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/{agent_id}/run")
async def run_agent(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await AgentService(db).debug_run(tenant_id, user.id, agent_id, dict(payload))


@router.post("/agents/import")
def import_agent(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentService(db).import_agent(tenant_id, user.id, dict(payload))


@router.get("/agents/{agent_id}/export")
def export_agent(agent_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("agents:read")), db: Session = Depends(get_db)) -> dict[str, object]:
    return AgentService(db).export_agent(tenant_id, user.id, agent_id)


@router.post("/agents/{agent_id}/disable")
def disable_agent(agent_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("agents:write")), db: Session = Depends(get_db)) -> ResourceRead:
    return ResourceRead.model_validate(ResourceService.to_dict(ResourceService(db).update("agents", tenant_id, user.id, agent_id, {"enabled": False, "status": "disabled"})))


@router.post("/agents/{agent_id}/enable")
def enable_agent(agent_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("agents:write")), db: Session = Depends(get_db)) -> ResourceRead:
    return ResourceRead.model_validate(ResourceService.to_dict(ResourceService(db).update("agents", tenant_id, user.id, agent_id, {"enabled": True, "status": "active"})))


add_action_route(router, method="post", path="/agent-templates", table="agents", permission="agents", action="create_template", output_table="agent_templates")

for path, table in [
    ("/agents/{agent_id}/versions", "agent_versions"),
    ("/agents/{agent_id}/runs", "agent_run_records"),
    ("/agent-templates", "agent_templates"),
]:
    add_list_route(router, method="get", path=path, table=table, permission="agents")

@router.get("/agents/{agent_id}/status")
def agent_status(agent_id: str, tenant_id: TenantIdDep, _: User = Depends(require_permission("agents:read")), db: Session = Depends(get_db)) -> dict[str, object]:
    return AgentService(db).status(tenant_id, agent_id)


@router.get("/agents/{agent_id}/versions/{version_id}", response_model=ResourceRead)
def agent_version_detail(
    agent_id: str,
    version_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> ResourceRead:
    row = ResourceService(db).get("agent_versions", tenant_id, version_id)
    if row.agent_id != agent_id:
        from app.core.constants import ErrorCode
        from app.core.errors import AppError

        raise AppError(ErrorCode.NOT_FOUND, "agent version not found", 404)
    return ResourceRead.model_validate(ResourceService.to_dict(row))


# --- Agent Publication (External API / Widget / Channel) ---

@router.post("/agents/{agent_id}/publish-api")
def publish_agent_api(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    data = dict(payload)
    data["agent_id"] = agent_id
    return AgentPublicationService(db).publish_as_api(tenant_id, user.id, data)


@router.post("/agents/{agent_id}/publish-widget")
def publish_agent_widget(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    data = dict(payload)
    data["agent_id"] = agent_id
    return AgentPublicationService(db).publish_as_widget(tenant_id, user.id, data)


@router.post("/agents/{agent_id}/publish-channel")
def publish_agent_channel(
    agent_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    data = dict(payload)
    data["agent_id"] = agent_id
    return AgentPublicationService(db).publish_as_channel(tenant_id, user.id, data)


@router.get("/agents/{agent_id}/publications")
def list_agent_publications(
    agent_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = AgentPublicationService(db).list_publications(tenant_id, 1, 100, {"agent_id": agent_id})
    return {"items": items, "total": total}


@router.get("/agent-publications")
def list_all_publications(
    tenant_id: TenantIdDep,
    page: int = 1,
    page_size: int = 20,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = AgentPublicationService(db).list_publications(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/agent-publications/{pub_id}")
def get_publication(
    pub_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentPublicationService(db).get_publication(tenant_id, pub_id)


@router.delete("/agent-publications/{pub_id}")
def delete_publication(
    pub_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return AgentPublicationService(db).delete_publication(tenant_id, user.id, pub_id)


@router.post("/agent-publications/{pub_id}/rotate-key")
def rotate_api_key(
    pub_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentPublicationService(db).rotate_api_key(tenant_id, user.id, pub_id)


@router.get("/agent-publications/{pub_id}/keys")
def list_api_keys(
    pub_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> list[dict[str, object]]:
    return AgentPublicationService(db).list_api_keys(tenant_id, pub_id)


@router.get("/agent-widgets")
def list_widgets(
    tenant_id: TenantIdDep,
    page: int = 1,
    page_size: int = 20,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = AgentPublicationService(db).list_widgets(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/agent-widgets/{widget_id}")
def get_widget(
    widget_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentPublicationService(db).get_widget(tenant_id, widget_id)


@router.put("/agent-widgets/{widget_id}")
def update_widget(
    widget_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentPublicationService(db).update_widget(tenant_id, user.id, widget_id, dict(payload))


@router.delete("/agent-widgets/{widget_id}")
def delete_widget(
    widget_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return AgentPublicationService(db).delete_widget(tenant_id, user.id, widget_id)


@router.get("/agent-channels")
def list_channels(
    tenant_id: TenantIdDep,
    page: int = 1,
    page_size: int = 20,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = AgentPublicationService(db).list_channels(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/agent-channels/{channel_id}")
def get_channel(
    channel_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentPublicationService(db).get_channel(tenant_id, channel_id)


@router.put("/agent-channels/{channel_id}")
def update_channel(
    channel_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AgentPublicationService(db).update_channel(tenant_id, user.id, channel_id, dict(payload))


@router.delete("/agent-channels/{channel_id}")
def delete_channel(
    channel_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return AgentPublicationService(db).delete_channel(tenant_id, user.id, channel_id)
