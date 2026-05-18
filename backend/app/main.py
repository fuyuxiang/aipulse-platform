from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.tracing import RequestContextMiddleware
from app.db.base import Base
from app.db.session import engine


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

    return app


app = create_app()

