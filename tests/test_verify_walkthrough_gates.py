"""Walkthrough gate verifier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_verify_walkthrough_gates_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_walkthrough_gates.py")],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
