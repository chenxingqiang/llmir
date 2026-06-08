#!/usr/bin/env python3
"""Generate all Nature-style LLMIR paper figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_ONLY = HERE / "paper-only"

SCRIPTS = [
    HERE / "create_architecture_diagram_nature.py",
    HERE / "create_block_size_chart_nature.py",
    HERE / "create_attention_speedup_nature.py",
    HERE / "create_mvp_evaluation_nature.py",
    PAPER_ONLY / "create_multi_model_comparison_nature.py",
    PAPER_ONLY / "create_prefix_cache_nature.py",
]


def main() -> int:
    for script in SCRIPTS:
        print(f"Running {script.name}...")
        subprocess.run([sys.executable, str(script)], cwd=str(script.parent), check=True)
    print("All Nature figures generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
