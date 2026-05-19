from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class MarketplaceService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def create_category(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        cat = self.resources.create("marketplace_categories", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"mcat-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "spec": {
                "description": str(payload.get("description", "")),
                "icon": str(payload.get("icon", "")),
                "sort_order": int(payload.get("sort_order", 0)),
                "parent_category_id": str(payload.get("parent_category_id", "")),
            },
        })
        return ResourceService.to_dict(cat)

    def list_categories(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("marketplace_categories", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def publish_listing(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        listing_type = str(payload.get("type", "agent"))
        resource_id = str(payload.get("resource_id", ""))
        listing = self.resources.create("marketplace_listings", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"ml-{uuid.uuid4().hex[:8]}",
            "status": "pending_review",
            "agent_id": resource_id if listing_type == "agent" else "",
            "spec": {
                "type": listing_type,
                "resource_id": resource_id,
                "description": str(payload.get("description", "")),
                "long_description": str(payload.get("long_description", "")),
                "category_id": str(payload.get("category_id", "")),
                "tags": payload.get("tags", []),
                "icon": str(payload.get("icon", "")),
                "screenshots": payload.get("screenshots", []),
                "version": str(payload.get("version", "1.0.0")),
                "pricing": payload.get("pricing", {"type": "free"}),
                "author": str(payload.get("author", "")),
                "author_tenant_id": tenant_id,
                "requirements": payload.get("requirements", []),
                "capabilities": payload.get("capabilities", []),
                "documentation_url": str(payload.get("documentation_url", "")),
                "source_url": str(payload.get("source_url", "")),
                "install_count": 0,
                "rating_avg": 0.0,
                "rating_count": 0,
                "published_at": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        return ResourceService.to_dict(listing)

    def update_listing(self, tenant_id: str, user_id: str, listing_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("marketplace_listings", tenant_id, user_id, listing_id, payload)
        return ResourceService.to_dict(row)

    def approve_listing(self, tenant_id: str, user_id: str, listing_id: str) -> dict[str, Any]:
        listing = self.resources.get("marketplace_listings", tenant_id, listing_id)
        spec = dict(listing.spec or {})
        spec["published_at"] = datetime.now(timezone.utc).isoformat()
        row = self.resources.update("marketplace_listings", tenant_id, user_id, listing_id, {
            "status": "published",
            "spec": spec,
        })
        return ResourceService.to_dict(row)

    def reject_listing(self, tenant_id: str, user_id: str, listing_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        listing = self.resources.get("marketplace_listings", tenant_id, listing_id)
        spec = dict(listing.spec or {})
        spec["rejection_reason"] = str(payload.get("reason", ""))
        row = self.resources.update("marketplace_listings", tenant_id, user_id, listing_id, {
            "status": "rejected",
            "spec": spec,
        })
        return ResourceService.to_dict(row)

    def unpublish_listing(self, tenant_id: str, user_id: str, listing_id: str) -> dict[str, Any]:
        row = self.resources.update("marketplace_listings", tenant_id, user_id, listing_id, {"status": "unpublished"})
        return ResourceService.to_dict(row)

    def delete_listing(self, tenant_id: str, user_id: str, listing_id: str) -> dict[str, str]:
        return self.resources.delete("marketplace_listings", tenant_id, user_id, listing_id)

    def get_listing(self, tenant_id: str, listing_id: str) -> dict[str, Any]:
        listing = ResourceService.to_dict(self.resources.get("marketplace_listings", tenant_id, listing_id))
        reviews, review_total = self.resources.list("marketplace_reviews", tenant_id, 1, 10, {"parent_id": listing_id})
        listing["reviews"] = [ResourceService.to_dict(r) for r in reviews]
        listing["review_count"] = review_total
        return listing

    def list_listings(self, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        effective_filters = dict(filters or {})
        effective_filters["status"] = "published"
        rows, total = self.resources.list("marketplace_listings", tenant_id, page, page_size, effective_filters)
        return [ResourceService.to_dict(row) for row in rows], total

    def search_listings(self, tenant_id: str, query: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("marketplace_listings", tenant_id, page, page_size, {"status": "published", "name": query})
        return [ResourceService.to_dict(row) for row in rows], total

    def install_listing(self, tenant_id: str, user_id: str, listing_id: str) -> dict[str, Any]:
        listing = self.resources.get("marketplace_listings", tenant_id, listing_id)
        spec = dict(listing.spec or {})
        listing_type = spec.get("type", "agent")
        resource_id = spec.get("resource_id", "")

        install = self.resources.create("marketplace_installs", tenant_id, user_id, {
            "name": f"install-{listing.name}",
            "code": f"mi-{uuid.uuid4().hex[:8]}",
            "status": "installed",
            "parent_id": listing_id,
            "agent_id": resource_id if listing_type == "agent" else "",
            "user_id": user_id,
            "spec": {
                "listing_id": listing_id,
                "listing_type": listing_type,
                "source_resource_id": resource_id,
                "source_tenant_id": spec.get("author_tenant_id", ""),
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "version": spec.get("version", "1.0.0"),
            },
        })

        spec["install_count"] = spec.get("install_count", 0) + 1
        self.resources.update("marketplace_listings", tenant_id, user_id, listing_id, {"spec": spec})

        cloned_resource = self._clone_resource(tenant_id, user_id, listing_type, resource_id, spec.get("author_tenant_id", tenant_id))

        return {
            "install_id": install.id,
            "listing_id": listing_id,
            "cloned_resource_id": cloned_resource.get("id", ""),
            "type": listing_type,
            "status": "installed",
        }

    def uninstall_listing(self, tenant_id: str, user_id: str, install_id: str) -> dict[str, str]:
        return self.resources.delete("marketplace_installs", tenant_id, user_id, install_id)

    def list_installs(self, tenant_id: str, user_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("marketplace_installs", tenant_id, page, page_size, {"user_id": user_id})
        return [ResourceService.to_dict(row) for row in rows], total

    def create_review(self, tenant_id: str, user_id: str, listing_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        rating = max(1, min(5, int(payload.get("rating", 5))))
        review = self.resources.create("marketplace_reviews", tenant_id, user_id, {
            "name": f"review-{uuid.uuid4().hex[:6]}",
            "code": f"mr-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "parent_id": listing_id,
            "user_id": user_id,
            "spec": {
                "rating": rating,
                "comment": str(payload.get("comment", "")),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        self._update_listing_rating(tenant_id, user_id, listing_id)
        return ResourceService.to_dict(review)

    def list_reviews(self, tenant_id: str, listing_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("marketplace_reviews", tenant_id, page, page_size, {"parent_id": listing_id})
        return [ResourceService.to_dict(row) for row in rows], total

    def _update_listing_rating(self, tenant_id: str, user_id: str, listing_id: str) -> None:
        reviews, total = self.resources.list("marketplace_reviews", tenant_id, 1, 1000, {"parent_id": listing_id})
        if not reviews:
            return
        ratings = [((r.spec or {}).get("rating", 5)) for r in reviews]
        avg = sum(ratings) / len(ratings)
        listing = self.resources.get("marketplace_listings", tenant_id, listing_id)
        spec = dict(listing.spec or {})
        spec["rating_avg"] = round(avg, 2)
        spec["rating_count"] = len(ratings)
        self.resources.update("marketplace_listings", tenant_id, user_id, listing_id, {"spec": spec})

    def _clone_resource(self, tenant_id: str, user_id: str, resource_type: str, resource_id: str, source_tenant_id: str) -> dict[str, Any]:
        table_map = {"agent": "agents", "tool": "tools", "workflow": "workflow_definitions", "prompt": "prompt_templates"}
        table = table_map.get(resource_type, "agents")
        try:
            source = self.resources.get(table, source_tenant_id, resource_id)
            cloned = self.resources.create(table, tenant_id, user_id, {
                "name": f"{source.name} (marketplace)",
                "code": f"clone-{uuid.uuid4().hex[:8]}",
                "status": "active",
                "spec": source.spec or {},
                "config": source.config or {},
            })
            return ResourceService.to_dict(cloned)
        except Exception:
            return {"id": "", "error": "clone_failed"}

    def get_stats(self, tenant_id: str) -> dict[str, Any]:
        _, total_listings = self.resources.list("marketplace_listings", tenant_id, 1, 1)
        _, total_installs = self.resources.list("marketplace_installs", tenant_id, 1, 1)
        _, total_reviews = self.resources.list("marketplace_reviews", tenant_id, 1, 1)
        return {
            "total_listings": total_listings,
            "total_installs": total_installs,
            "total_reviews": total_reviews,
        }
