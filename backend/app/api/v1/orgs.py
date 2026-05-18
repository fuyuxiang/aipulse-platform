from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_crud_routes

router = APIRouter(tags=["organizations"])
add_crud_routes(router, table="organizations", prefix="/orgs", permission="orgs")
add_crud_routes(router, table="organization_members", prefix="/organization-members", permission="orgs")

