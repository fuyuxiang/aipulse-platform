from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_crud_routes

router = APIRouter(tags=["permissions"])
add_crud_routes(router, table="permissions", prefix="/permissions", permission="permissions")

