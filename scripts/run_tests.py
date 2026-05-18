from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(["pytest"], cwd=ROOT / "backend", check=True)
    subprocess.run(["pytest"], cwd=ROOT / "runtime", check=True)
    subprocess.run(["npm", "run", "test"], cwd=ROOT / "frontend", check=True)


if __name__ == "__main__":
    main()

