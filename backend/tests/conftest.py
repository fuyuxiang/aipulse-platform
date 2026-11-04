from __future__ import annotations

import os
from pathlib import Path
import tempfile

TEST_ROOT = Path(tempfile.mkdtemp(prefix="aipulse-platform-tests-", dir="/private/tmp"))
TEST_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH = TEST_ROOT / "aipulse-test.db"
if DB_PATH.exists():
    DB_PATH.unlink()

os.environ["AIPULSE_DATABASE_URL"] = f"sqlite:///{DB_PATH}"
os.environ["AIPULSE_DATA_DIR"] = str(TEST_ROOT / "data")
os.environ["AIPULSE_OBJECT_STORE_DIR"] = str(TEST_ROOT / "files")
os.environ["AIPULSE_VECTOR_STORE_DIR"] = str(TEST_ROOT / "vectors")
os.environ["AIPULSE_LOG_PATH"] = str(TEST_ROOT / "logs" / "backend.jsonl")
os.environ["AIPULSE_TRACE_PATH"] = str(TEST_ROOT / "traces" / "backend-traces.jsonl")

from app.db.base import Base  # noqa: E402
from app.db.init_data import initialize_defaults  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402

Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    initialize_defaults(db)
