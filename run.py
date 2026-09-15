"""Launch the T5 summarization Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Start Streamlit with the application's main entry point."""
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "app" / "main.py"

    if not app_path.is_file():
        raise FileNotFoundError(f"Streamlit entry point not found: {app_path}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())