"""E8 lab smoke script (honest CPU skip)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts/e8_lab_smoke.sh"
STRICT = ROOT / "scripts/e8_lab_run.sh"


def test_e8_lab_scripts_exist():
    assert SMOKE.is_file()
    assert STRICT.is_file()
    assert "status=completed" in STRICT.read_text(encoding="utf-8")


def test_e8_lab_smoke_writes_json_and_exits_zero():
    proc = subprocess.run(
        ["bash", str(SMOKE)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    out = ROOT / "IEEE-conference/benchmarks/e8_empirical_gpu.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["experiment"] == "E8"
    assert data["status"] in ("skipped", "completed")
