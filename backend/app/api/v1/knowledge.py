from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes
from app.core.response import ListResponse
from app.models.core import User
from app.schemas.common import ResourceRead
from app.services.knowledge_service import KnowledgeService
from app.services.resource_service import ResourceService

router = APIRouter(tags=["knowledge"])

add_crud_routes(router, table="knowledge_bases", prefix="/knowledge-bases", permission="knowledge")

add_action_route(router, method="post", path="/knowledge-bases/{kb_id}/permissions", table="knowledge_bases", permission="knowledge", action="permissions", output_table="knowledge_base_permissions")


@router.post("/knowledge-bases/{kb_id}/documents")
def upload_document(
    kb_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("knowledge:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return KnowledgeService(db).upload_document(tenant_id, user.id, kb_id, dict(payload))


@router.get("/knowledge-bases/{kb_id}/documents", response_model=ListResponse[ResourceRead])
def list_documents(
    kb_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("knowledge:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("knowledge_documents", tenant_id, page, page_size, {"knowledge_base_id": kb_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/knowledge-documents/{document_id}", response_model=ResourceRead)
def get_document(document_id: str, tenant_id: TenantIdDep, _: User = Depends(require_permission("knowledge:read")), db: Session = Depends(get_db)) -> ResourceRead:
    return ResourceRead.model_validate(ResourceService.to_dict(ResourceService(db).get("knowledge_documents", tenant_id, document_id)))


@router.delete("/knowledge-documents/{document_id}")
def delete_document(document_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("knowledge:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return KnowledgeService(db).delete_document(tenant_id, user.id, document_id)


@router.post("/knowledge-documents/{document_id}/parse")
def parse_document(document_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("knowledge:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return KnowledgeService(db).parse_document(tenant_id, user.id, document_id)


@router.get("/knowledge-documents/{document_id}/parse-status", response_model=ListResponse[ResourceRead])
def parse_status(
    document_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("knowledge:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("knowledge_parse_jobs", tenant_id, page, page_size, {"parent_id": document_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/knowledge-documents/{document_id}/chunks", response_model=ListResponse[ResourceRead])
def document_chunks(
    document_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("knowledge:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("knowledge_chunks", tenant_id, page, page_size, {"parent_id": document_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.post("/knowledge-documents/{document_id}/reindex")
def reindex_document(document_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("knowledge:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return KnowledgeService(db).reindex_document(tenant_id, user.id, document_id)


@router.post("/knowledge-bases/{kb_id}/rebuild-index")
def rebuild_index(kb_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("knowledge:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return KnowledgeService(db).rebuild_index(tenant_id, user.id, kb_id)


@router.post("/knowledge-bases/{kb_id}/retrieve")
def retrieve(
    kb_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("knowledge:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return KnowledgeService(db).retrieve(tenant_id, user.id, kb_id, dict(payload))


@router.post("/knowledge-bases/{kb_id}/rerank")
def rerank(
    kb_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("knowledge:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return KnowledgeService(db).rerank(tenant_id, user.id, kb_id, dict(payload))


@router.get("/knowledge-bases/{kb_id}/retrieval-logs", response_model=ListResponse[ResourceRead])
def retrieval_logs(
    kb_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("knowledge:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("knowledge_retrieval_logs", tenant_id, page, page_size, {"knowledge_base_id": kb_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/knowledge-bases/{kb_id}/stats")
def knowledge_stats(kb_id: str, tenant_id: TenantIdDep, _: User = Depends(require_permission("knowledge:read")), db: Session = Depends(get_db)) -> dict[str, object]:
    return KnowledgeService(db).stats(tenant_id, kb_id)
