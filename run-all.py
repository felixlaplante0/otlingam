import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = (
    ("scripts/statistical-performance.py", "--nd"),
    ("scripts/statistical-performance.py", "--heterogeneity"),
    ("scripts/statistical-performance.py", "--k"),
    ("scripts/runtime-scaling.py",),
)


for command in COMMANDS:
    subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
