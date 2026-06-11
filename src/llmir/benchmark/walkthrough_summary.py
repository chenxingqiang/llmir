"""Reviewer-facing walkthrough summary from committed artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from llmir.benchmark.artifact_bundle import verify_artifact_bundle
from llmir.compiler.mlir_lit_suite import default_lit_dir, run_lit_suite
from llmir.compiler.opt_driver import find_mlir_opt


def _repo_root(start: Optional[Path] = None) -> Path:
    if start is not None:
        return start
    return Path(__file__).resolve().parents[3]


def build_walkthrough_summary(root: Optional[Path] = None) -> Dict[str, Any]:
    root = _repo_root(root)
    m6 = verify_artifact_bundle(root, check_figures=False)
    lit = run_lit_suite(default_lit_dir(root))

    e8_path = root / "IEEE-conference/benchmarks/e8_empirical_gpu.json"
    e8: Dict[str, Any] = {}
    if e8_path.is_file():
        e8 = json.loads(e8_path.read_text(encoding="utf-8"))

    failed = [a for a in m6.artifacts if not a["ok"]]

    return {
        "mode": "a_class_walkthrough_summary",
        "evidence_class": "A",
        "m6_all_pass": m6.all_pass,
        "artifact_count": len(m6.artifacts),
        "artifacts_failed": failed,
        "e8_status": e8.get("status", "missing"),
        "e8_evidence_class": e8.get("evidence_class", "B"),
        "mlir_opt_available": find_mlir_opt() is not None,
        "mlir_lit_suite_status": lit["status"],
        "mlir_lit_passed": lit["passed"],
        "mlir_lit_skipped": lit["skipped"],
        "reproduce_command": "bash scripts/reproduce_paper.sh",
        "walkthrough_command": "bash scripts/walkthrough_a_class.sh",
    }
