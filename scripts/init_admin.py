from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db.init_data import initialize_defaults
from app.db.session import SessionLocal


def main() -> None:
    with SessionLocal() as db:
        ids = initialize_defaults(db)
    print("default admin ready: tenant=default username=admin password=admin123456")
    print(f"tenant_id={ids['tenant_id']} user_id={ids['user_id']}")


if __name__ == "__main__":
    main()

