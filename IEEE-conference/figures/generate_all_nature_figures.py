#!/usr/bin/env python3
"""Generate CI-verified / measured Nature-style paper figures (default)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Measured or E1–E3-verified panels only. For illustrative sweeps/heatmaps, use
# generate_projected_figures.py
SCRIPTS = [
    HERE / "create_architecture_diagram_nature.py",
    HERE / "create_e1_e3_evaluation_nature.py",
    HERE / "create_measured_figures_nature.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"Running {script.name}...")
        subprocess.run([sys.executable, str(script)], cwd=str(script.parent), check=True)
    print("Measured Nature figures generated.")
    print("For projected figures: python3 generate_projected_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
