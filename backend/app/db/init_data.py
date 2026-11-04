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
    "chat",
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


def initialize_tenant_rbac(db: Session, tenant_id: str, actor_id: str) -> None:
    role = db.scalar(select(Role).where(Role.tenant_id == tenant_id, Role.code == "admin", Role.deleted_at.is_(None)))
    if role is None:
        role = Role(tenant_id=tenant_id, code="admin", name="租户管理员", created_by=actor_id, updated_by=actor_id)
        db.add(role)
        db.flush()
    for domain in DOMAINS:
        for action in (["read", "write", "approve", "*"] if domain in {"tools", "workflows"} else ["read", "write", "*"]):
            code = f"{domain}:{action}"
            permission = db.scalar(select(Permission).where(Permission.tenant_id == tenant_id, Permission.code == code))
            if permission is None:
                permission = Permission(tenant_id=tenant_id, code=code, name=code, resource_type=domain, action=action, created_by=actor_id, updated_by=actor_id)
                db.add(permission)
                db.flush()
            if db.scalar(select(RolePermission).where(RolePermission.tenant_id == tenant_id, RolePermission.role_id == role.id, RolePermission.permission_id == permission.id)) is None:
                db.add(RolePermission(tenant_id=tenant_id, role_id=role.id, permission_id=permission.id, created_by=actor_id, updated_by=actor_id))


def initialize_defaults(db: Session) -> dict[str, str]:
    ids = ensure_default_identity(db)
    tenant = db.get(Tenant, ids["tenant_id"])
    assert tenant is not None
    for tenant_id in db.scalars(select(Tenant.id).where(Tenant.deleted_at.is_(None))):
        initialize_tenant_rbac(db, tenant_id, ids["user_id"])
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
