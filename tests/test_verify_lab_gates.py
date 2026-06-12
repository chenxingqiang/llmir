"""Lab gate verifier."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_lab_gates.py"


def test_verify_lab_gates_script_passes():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Lab gates OK" in proc.stdout


def test_verify_lab_gates_module_accepts_live_summary():
    # Import via subprocess path used by verify_walkthrough_gates
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); "
            "from verify_lab_gates import verify_lab_gates; "
            "e, n = verify_lab_gates(); assert not e",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
