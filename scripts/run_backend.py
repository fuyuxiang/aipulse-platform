from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'backend'}:{ROOT}:{ROOT / 'echo-agent'}:{env.get('PYTHONPATH', '')}"
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT / "backend", env=env, check=True)


if __name__ == "__main__":
    main()

