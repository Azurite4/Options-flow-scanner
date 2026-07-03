"""Launcher: run this file (F5 / Run button in VS Code) to start the GUI."""

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "src" / "app.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP)],
        check=False,
    )