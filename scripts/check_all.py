from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    run(["ruff", "check", "."], ROOT / "backend")
    run(["mypy", "app"], ROOT / "backend")
    run(["pytest"], ROOT / "backend")
    run(["pytest"], ROOT / "runtime")
    run(["npm", "run", "lint"], ROOT / "frontend")
    run(["npm", "run", "test"], ROOT / "frontend")
    run(["npm", "run", "build"], ROOT / "frontend")


if __name__ == "__main__":
    main()

