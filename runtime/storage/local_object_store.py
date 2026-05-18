from __future__ import annotations

import hashlib
from pathlib import Path


class LocalObjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, tenant_id: str, name: str, data: bytes) -> dict[str, object]:
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / tenant_id / digest[:2] / f"{digest}-{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"uri": str(path), "sha256": digest, "size": len(data)}

    def read_bytes(self, uri: str) -> bytes:
        return Path(uri).read_bytes()

