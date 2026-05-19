from __future__ import annotations

import asyncio
import contextlib
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.tracing import RequestContextMiddleware
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models.core import Tenant
from app.services.scheduler_service import SchedulerService


def create_app() -> FastAPI:
    if str(settings.project_root) not in sys.path:
        sys.path.insert(0, str(settings.project_root))
    configure_logging()
    for path in ["sqlite", "files", "vector", "logs", "traces", "exports"]:
        (settings.resolved_data_dir / path).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    app = FastAPI(title=settings.app_name, version="0.1.0", openapi_url=f"{settings.api_prefix}/openapi.json")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    scheduler_task: asyncio.Task[None] | None = None

    @app.on_event("startup")
    async def start_scheduler() -> None:
        nonlocal scheduler_task
        if settings.scheduler_enabled and scheduler_task is None:
            scheduler_task = asyncio.create_task(_scheduler_loop())

    @app.on_event("shutdown")
    async def stop_scheduler() -> None:
        nonlocal scheduler_task
        if scheduler_task is not None:
            scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scheduler_task
            scheduler_task = None

    return app


app = create_app()


async def _scheduler_loop() -> None:
    while True:
        await asyncio.sleep(max(5, int(settings.scheduler_poll_seconds)))
        with SessionLocal() as db:
            tenant_ids = list(db.scalars(select(Tenant.id).where(Tenant.status == "active", Tenant.deleted_at.is_(None))).all())
            service = SchedulerService(db)
            for tenant_id in tenant_ids:
                try:
                    await service.run_due_jobs(tenant_id, "system")
                except Exception:
                    continue
