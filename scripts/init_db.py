from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db.base import Base  # noqa: E402
from app.db.init_data import initialize_defaults  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ids = initialize_defaults(db)
    print(f"initialized database tenant_id={ids['tenant_id']} admin_user_id={ids['user_id']}")


if __name__ == "__main__":
    main()

