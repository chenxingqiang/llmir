#!/usr/bin/env python3
"""Aggregate lab smoke JSON artifacts into one reviewer snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.pypi_release_status import read_local_version  # noqa: E402

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/lab_status_summary.json"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_lab_status_summary(root: Path) -> dict:
    mlir = _load_json(root / "IEEE-conference/benchmarks/mlir_lit_suite_status.json")
    e8 = _load_json(root / "IEEE-conference/benchmarks/e8_empirical_gpu.json")
    pypi = _load_json(root / "IEEE-conference/benchmarks/pypi_release_status.json")

    prereq_proc = subprocess.run(
        ["bash", str(root / "scripts/check_native_build_prereqs.sh")],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    prereq_text = (prereq_proc.stdout or "") + (prereq_proc.stderr or "")
    llvm_ready = "llvm-config" in prereq_text and "MISS llvm" not in prereq_text

    return {
        "mode": "lab_status_summary",
        "package_version": read_local_version(root),
        "mlir_lit_status": (mlir or {}).get("status", "missing"),
        "mlir_lit_passed": (mlir or {}).get("passed", 0),
        "e8_status": (e8 or {}).get("status", "missing"),
        "pypi_release_status": (pypi or {}).get("status", "missing"),
        "pypi_published": (pypi or {}).get("published", False),
        "native_build_prereqs_ok": llvm_ready and prereq_proc.returncode == 0,
        "commands": {
            "lab_smoke_all": "bash scripts/lab_smoke_all.sh",
            "mlir_lit": "bash scripts/mlir_lit_smoke.sh",
            "e8_gpu": "bash scripts/e8_lab_smoke.sh",
            "native_build": "bash scripts/check_native_build_prereqs.sh",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write lab status summary JSON")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = parser.parse_args()

    summary = build_lab_status_summary(ROOT)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.json_out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
