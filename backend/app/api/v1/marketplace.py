from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.models.core import User
from app.services.marketplace_service import MarketplaceService

router = APIRouter(tags=["marketplace"])


@router.post("/marketplace/categories")
def create_category(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).create_category(tenant_id, user.id, dict(payload))


@router.get("/marketplace/categories")
def list_categories(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = MarketplaceService(db).list_categories(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/marketplace/listings")
def publish_listing(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).publish_listing(tenant_id, user.id, dict(payload))


@router.get("/marketplace/listings")
def list_listings(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    category_id: str = "",
    listing_type: str = "",
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    filters: dict[str, object] = {}
    if category_id:
        filters["category_id"] = category_id
    if listing_type:
        filters["type"] = listing_type
    items, total = MarketplaceService(db).list_listings(tenant_id, page, page_size, filters or None)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/marketplace/search")
def search_listings(
    tenant_id: TenantIdDep,
    q: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = MarketplaceService(db).search_listings(tenant_id, q, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size, "query": q}


@router.get("/marketplace/listings/{listing_id}")
def get_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).get_listing(tenant_id, listing_id)


@router.put("/marketplace/listings/{listing_id}")
def update_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).update_listing(tenant_id, user.id, listing_id, dict(payload))


@router.post("/marketplace/listings/{listing_id}/approve")
def approve_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).approve_listing(tenant_id, user.id, listing_id)


@router.post("/marketplace/listings/{listing_id}/reject")
def reject_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).reject_listing(tenant_id, user.id, listing_id, dict(payload))


@router.post("/marketplace/listings/{listing_id}/unpublish")
def unpublish_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).unpublish_listing(tenant_id, user.id, listing_id)


@router.delete("/marketplace/listings/{listing_id}")
def delete_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return MarketplaceService(db).delete_listing(tenant_id, user.id, listing_id)


@router.post("/marketplace/listings/{listing_id}/install")
def install_listing(
    listing_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).install_listing(tenant_id, user.id, listing_id)


@router.get("/marketplace/installs")
def list_installs(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    user: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = MarketplaceService(db).list_installs(tenant_id, user.id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.delete("/marketplace/installs/{install_id}")
def uninstall_listing(
    install_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return MarketplaceService(db).uninstall_listing(tenant_id, user.id, install_id)


@router.post("/marketplace/listings/{listing_id}/reviews")
def create_review(
    listing_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).create_review(tenant_id, user.id, listing_id, dict(payload))


@router.get("/marketplace/listings/{listing_id}/reviews")
def list_reviews(
    listing_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = MarketplaceService(db).list_reviews(tenant_id, listing_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/marketplace/stats")
def marketplace_stats(
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MarketplaceService(db).get_stats(tenant_id)
