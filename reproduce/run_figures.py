"""Run contained figure workflows; report Fig. 4's licensed dependency clearly."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

for script in ("figure2.py", "figure3.py", "figure4.py", "figure5.py", "figure6.py"):
    print()
    print(f"== {script} ==")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
    if script == "figure4.py" and result.returncode == 2:
        print("Fig. 4: EXTERNAL_DEPENDENCY (HydroBASINS v1c Asia level 9)")
        continue
    if result.returncode:
        raise SystemExit(result.returncode)

print()
print("Figure workflows complete; Fig. 4 external dependency is documented above.")
