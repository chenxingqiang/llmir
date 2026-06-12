"""E8 lab smoke script (honest CPU skip)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts/e8_lab_smoke.sh"
STRICT = ROOT / "scripts/e8_lab_run.sh"
VERIFY = ROOT / "scripts/verify_e8_lab.py"
PREFLIGHT = ROOT / "scripts/e8_lab_preflight.sh"


def test_e8_lab_scripts_exist():
    assert SMOKE.is_file()
    assert STRICT.is_file()
    assert PREFLIGHT.is_file()
    assert VERIFY.is_file()
    assert "verify_e8_lab.py" in STRICT.read_text(encoding="utf-8")


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


def test_e8_lab_preflight_runs_without_strict():
    proc = subprocess.run(
        ["bash", str(PREFLIGHT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Preflight OK" in proc.stdout


def test_verify_e8_lab_accepts_skipped_json(tmp_path):
    payload = {
        "experiment": "E8",
        "evidence_class": "B",
        "status": "skipped",
        "results": [],
        "reason": "no_cuda_available",
    }
    path = tmp_path / "e8.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("verify_e8_lab", VERIFY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with patch("sys.argv", ["verify", "--json", str(path)]):
        assert module.main() == 0


def test_verify_e8_lab_require_completed_fails_when_skipped(tmp_path):
    payload = {
        "experiment": "E8",
        "evidence_class": "B",
        "status": "skipped",
        "results": [],
    }
    path = tmp_path / "e8.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("verify_e8_lab", VERIFY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with patch("sys.argv", ["verify", "--json", str(path), "--require-completed"]):
        assert module.main() == 1
