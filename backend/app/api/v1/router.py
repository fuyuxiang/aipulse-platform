from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import (
    agents,
    alerts,
    audit,
    auth,
    evaluation,
    knowledge,
    memory,
    model_management,
    model_routing,
    observability,
    orgs,
    permissions,
    roles,
    runtime,
    security,
    system,
    tenants,
    tools_center,
    users,
    workflows,
)

api_router = APIRouter()

for module in [
    auth,
    tenants,
    users,
    orgs,
    roles,
    permissions,
    agents,
    workflows,
    model_management,
    model_routing,
    tools_center,
    knowledge,
    memory,
    observability,
    audit,
    security,
    evaluation,
    alerts,
    runtime,
    system,
]:
    api_router.include_router(module.router)

