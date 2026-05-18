from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all platform tables."""


from app.models.core import *  # noqa: E402,F403
from app.models.resources import *  # noqa: E402,F403
