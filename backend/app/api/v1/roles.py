from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_crud_routes

router = APIRouter(tags=["roles"])
add_crud_routes(router, table="roles", prefix="/roles", permission="roles")
add_crud_routes(router, table="role_permissions", prefix="/role-permissions", permission="roles")
add_crud_routes(router, table="user_roles", prefix="/user-roles", permission="roles")

