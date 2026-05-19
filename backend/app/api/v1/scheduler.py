from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.models.core import User
from app.services.scheduler_service import SchedulerService

router = APIRouter(tags=["scheduler"])


@router.post("/scheduler/jobs")
def create_job(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).create_job(tenant_id, user.id, dict(payload))


@router.get("/scheduler/jobs")
def list_jobs(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    job_type: str = "",
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    filters = {"job_type": job_type} if job_type else None
    items, total = SchedulerService(db).list_jobs(tenant_id, page, page_size, filters)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/scheduler/jobs/{job_id}")
def get_job(
    job_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).get_job(tenant_id, job_id)


@router.put("/scheduler/jobs/{job_id}")
def update_job(
    job_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).update_job(tenant_id, user.id, job_id, dict(payload))


@router.delete("/scheduler/jobs/{job_id}")
def delete_job(
    job_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return SchedulerService(db).delete_job(tenant_id, user.id, job_id)


@router.post("/scheduler/jobs/{job_id}/enable")
def enable_job(
    job_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).enable_job(tenant_id, user.id, job_id)


@router.post("/scheduler/jobs/{job_id}/disable")
def disable_job(
    job_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).disable_job(tenant_id, user.id, job_id)


@router.post("/scheduler/jobs/{job_id}/trigger")
async def trigger_job(
    job_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await SchedulerService(db).trigger_job(tenant_id, user.id, job_id, dict(payload) or None)


@router.get("/scheduler/jobs/{job_id}/executions")
def list_executions(
    job_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = SchedulerService(db).list_executions(tenant_id, job_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/scheduler/webhooks")
def create_webhook(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).create_webhook(tenant_id, user.id, dict(payload))


@router.get("/scheduler/webhooks")
def list_webhooks(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = SchedulerService(db).list_webhooks(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/scheduler/webhooks/{webhook_id}")
def get_webhook(
    webhook_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).get_webhook(tenant_id, webhook_id)


@router.put("/scheduler/webhooks/{webhook_id}")
def update_webhook(
    webhook_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).update_webhook(tenant_id, user.id, webhook_id, dict(payload))


@router.delete("/scheduler/webhooks/{webhook_id}")
def delete_webhook(
    webhook_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return SchedulerService(db).delete_webhook(tenant_id, user.id, webhook_id)


@router.post("/scheduler/webhooks/{webhook_id}/invoke")
async def invoke_webhook(
    webhook_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    _: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await SchedulerService(db).handle_webhook(tenant_id, webhook_id, dict(payload))


@router.post("/scheduler/triggers")
def create_trigger(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).create_trigger(tenant_id, user.id, dict(payload))


@router.get("/scheduler/triggers")
def list_triggers(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = SchedulerService(db).list_triggers(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.put("/scheduler/triggers/{trigger_id}")
def update_trigger(
    trigger_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).update_trigger(tenant_id, user.id, trigger_id, dict(payload))


@router.delete("/scheduler/triggers/{trigger_id}")
def delete_trigger(
    trigger_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("runtime:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return SchedulerService(db).delete_trigger(tenant_id, user.id, trigger_id)


@router.get("/scheduler/stats")
def scheduler_stats(
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("runtime:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return SchedulerService(db).get_stats(tenant_id)
