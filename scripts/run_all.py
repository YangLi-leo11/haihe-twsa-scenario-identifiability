"""Run every contained publication reproduction workflow."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(path: Path) -> None:
    print()
    print(f"== {path.relative_to(ROOT)} ==")
    subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    run(ROOT / "reproduce/run_figures.py")
    run(ROOT / "reproduce/run_core_analysis.py")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)
    print()
    print("All contained workflows and regression tests passed.")
