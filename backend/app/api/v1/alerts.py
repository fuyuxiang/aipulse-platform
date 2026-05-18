from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_crud_routes

router = APIRouter(tags=["alerts"])
add_crud_routes(router, table="alert_rules", prefix="/alerts/rules", permission="alerts")
add_crud_routes(router, table="alert_events", prefix="/alerts/events", permission="alerts")

