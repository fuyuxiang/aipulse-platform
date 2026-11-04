from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    run([sys.executable, "-m", "ruff", "check", "backend", "runtime", "scripts"], ROOT)
    run([sys.executable, "-m", "mypy", "app"], ROOT / "backend")
    run([sys.executable, "-m", "pytest"], ROOT / "backend")
    run([sys.executable, "-m", "pytest"], ROOT / "runtime")
    run(["npm", "run", "lint"], ROOT / "frontend")
    run(["npm", "run", "test"], ROOT / "frontend")
    run(["npm", "run", "build"], ROOT / "frontend")


if __name__ == "__main__":
    main()

