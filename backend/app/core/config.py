from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local-source configuration with production replacement points."""

    model_config = SettingsConfigDict(env_prefix="AIPULSE_", env_file=".env", extra="ignore")

    app_name: str = "AIPulse Platform"
    environment: str = "local"
    api_prefix: str = "/api/v1"
    jwt_secret: str = Field(default_factory=lambda: os.getenv("AIPULSE_JWT_SECRET", "local-dev-secret-change-me"))
    jwt_issuer: str = "aipulse-platform"
    access_token_minutes: int = 60
    refresh_token_days: int = 14
    password_salt: str = Field(default_factory=lambda: os.getenv("AIPULSE_PASSWORD_SALT", "local-dev-password-salt"))
    echo_agent_path: Path = Field(default_factory=lambda: Path("../echo-agent"))
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    database_url: str = "sqlite:///data/sqlite/aipulse.db"
    log_path: Path = Field(default_factory=lambda: Path("data/logs/backend.jsonl"))
    trace_path: Path = Field(default_factory=lambda: Path("data/traces/backend-traces.jsonl"))
    object_store_dir: Path = Field(default_factory=lambda: Path("data/files"))
    vector_store_dir: Path = Field(default_factory=lambda: Path("data/vector"))
    cors_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:3000", "http://localhost:3000"])

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else (self.project_root / value).resolve()

    @property
    def resolved_data_dir(self) -> Path:
        return self.resolve_path(self.data_dir)

    @property
    def resolved_echo_agent_path(self) -> Path:
        return self.resolve_path(self.echo_agent_path)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.database_url.startswith("sqlite:///") and not settings.database_url.startswith("sqlite:////"):
        raw_path = Path(settings.database_url.removeprefix("sqlite:///"))
        absolute = raw_path if raw_path.is_absolute() else settings.project_root / raw_path
        settings.database_url = f"sqlite:///{absolute}"
    settings.resolved_data_dir.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
