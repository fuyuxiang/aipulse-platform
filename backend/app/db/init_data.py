from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.constants import MODEL_PROVIDER_TYPES, MODEL_TYPES, ROUTING_POLICY_TYPES
from app.models.core import Permission, Role, RolePermission, Tenant
from app.services._shared.auth_service import ensure_default_identity
from app.services._shared.resource_service import ResourceService


DOMAINS = [
    "tenants",
    "users",
    "orgs",
    "roles",
    "permissions",
    "agents",
    "workflows",
    "models",
    "model-routing",
    "tools",
    "knowledge",
    "memory",
    "observability",
    "audit",
    "security",
    "evaluation",
    "alerts",
    "runtime",
    "system",
]


def initialize_defaults(db: Session) -> dict[str, str]:
    ids = ensure_default_identity(db)
    tenant = db.get(Tenant, ids["tenant_id"])
    assert tenant is not None
    admin_role = db.scalar(select(Role).where(Role.tenant_id == tenant.id, Role.code == "admin"))
    assert admin_role is not None
    for domain in DOMAINS:
        for action in ["read", "write", "*"]:
            code = f"{domain}:{action}"
            permission = db.scalar(select(Permission).where(Permission.tenant_id == tenant.id, Permission.code == code))  # type: ignore[arg-type]
            if permission is None:
                permission = Permission(
                    tenant_id=tenant.id,
                    code=code,
                    name=code,
                    permission_type="api",
                    resource_type=domain,
                    action=action,
                    created_by=ids["user_id"],
                    updated_by=ids["user_id"],
                )
                db.add(permission)
                db.flush()
            if db.scalar(select(RolePermission).where(RolePermission.tenant_id == tenant.id, RolePermission.role_id == admin_role.id, RolePermission.permission_id == permission.id)) is None:  # type: ignore[arg-type]
                db.add(RolePermission(tenant_id=tenant.id, role_id=admin_role.id, permission_id=permission.id, created_by=ids["user_id"], updated_by=ids["user_id"]))
    db.commit()
    service = ResourceService(db)
    provider_rows, provider_total = service.list("model_providers", tenant.id, 1, 1, {"code": "echo-agent-native"})
    if provider_total == 0:
        provider = service.create(
            "model_providers",
            tenant.id,
            ids["user_id"],
            {
                "name": "Echo Agent Native",
                "code": "echo-agent-native",
                "provider_type": "echo_agent_native",
                "status": "active",
                "enabled": True,
                "config": {"provider_types": sorted(MODEL_PROVIDER_TYPES)},
            },
        )
        for model_type in sorted(MODEL_TYPES):
            service.create(
                "models",
                tenant.id,
                ids["user_id"],
                {
                    "name": f"Local {model_type}",
                    "code": f"local-{model_type}",
                    "provider_id": provider.id,
                    "provider_type": "echo_agent_native",
                    "model_type": model_type,
                    "model_id": f"local-{model_type}",
                    "status": "active",
                    "enabled": True,
                    "config": {"embedding_dimensions": 128, "routing_policy_types": sorted(ROUTING_POLICY_TYPES)},
                },
            )
    return ids
