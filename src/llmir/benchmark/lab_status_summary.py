"""Aggregate optional lab smoke JSON into one reviewer snapshot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, cast

from llmir.benchmark.pypi_release_status import read_local_version


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def native_build_prereqs_ok(root: Path) -> bool:
    proc = subprocess.run(
        ["bash", str(root / "scripts/check_native_build_prereqs.sh")],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    llvm_ok = "llvm-config" in text and "MISS llvm" not in text
    return llvm_ok and proc.returncode == 0


def build_lab_status_summary(root: Path) -> Dict[str, Any]:
    mlir = _load_json(root / "IEEE-conference/benchmarks/mlir_lit_suite_status.json")
    e8 = _load_json(root / "IEEE-conference/benchmarks/e8_empirical_gpu.json")
    pypi = _load_json(root / "IEEE-conference/benchmarks/pypi_release_status.json")

    return {
        "mode": "lab_status_summary",
        "package_version": read_local_version(root),
        "mlir_lit_status": (mlir or {}).get("status", "missing"),
        "mlir_lit_passed": (mlir or {}).get("passed", 0),
        "e8_status": (e8 or {}).get("status", "missing"),
        "pypi_release_status": (pypi or {}).get("status", "missing"),
        "pypi_published": (pypi or {}).get("published", False),
        "native_build_prereqs_ok": native_build_prereqs_ok(root),
        "commands": {
            "lab_smoke_all": "bash scripts/lab_smoke_all.sh",
            "mlir_lit_preflight": "bash scripts/mlir_lit_preflight.sh",
            "mlir_lit": "bash scripts/mlir_lit_smoke.sh",
            "mlir_lit_verify": "python3 scripts/verify_mlir_lit_suite.py",
            "mlir_lit_strict": "mlir-lit-lab.yml require_passed=true",
            "e8_preflight": "bash scripts/e8_lab_preflight.sh",
            "e8_gpu": "bash scripts/e8_lab_smoke.sh",
            "e8_verify": "python3 scripts/verify_e8_lab.py",
            "e8_strict": "e8-gpu-lab.yml require_completed=true",
            "pypi_verify": "python3 scripts/verify_pypi_release.py",
            "pypi_republish_preflight": "bash scripts/pypi_republish_preflight.sh vX.Y.Z",
            "pypi_strict": "pypi-republish.yml + --require-published",
            "native_build": "bash scripts/check_native_build_prereqs.sh",
            "native_strict": "native-runtime.yml strict_prereqs=true",
            "lab_gates": "python3 scripts/verify_lab_gates.py",
        },
    }
