"""Run contained numerical checks from the repository root."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/run_qc.py"), run_name="__main__")
