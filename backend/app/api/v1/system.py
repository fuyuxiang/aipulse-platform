from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_crud_routes

router = APIRouter(tags=["system"])
add_crud_routes(router, table="system_configs", prefix="/system/configs", permission="system")

