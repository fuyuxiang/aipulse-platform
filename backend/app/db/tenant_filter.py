from __future__ import annotations

from typing import Any

from sqlalchemy import Select


def apply_tenant_filter(stmt: Select[Any], model: type[Any], tenant_id: str, *, cross_tenant: bool = False) -> Select[Any]:
    if cross_tenant or not hasattr(model, "tenant_id"):
        return stmt
    return stmt.where(model.tenant_id == tenant_id)

