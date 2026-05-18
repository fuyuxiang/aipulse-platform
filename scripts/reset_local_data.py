from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    for name in ["sqlite", "files", "vector", "logs", "traces", "exports"]:
        path = DATA / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    print("local data reset")


if __name__ == "__main__":
    main()

