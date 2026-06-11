"""Discover and optionally execute MLIR lit tests under test/Dialect/LLM/."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llmir.compiler.opt_driver import find_mlir_opt

RUN_LINE_RE = re.compile(r"^\s*//\s*RUN:\s*(.+)$", re.MULTILINE)

# Tier-A lit files exercised by the Python suite (excludes aspirational llmir-opt demos).
LIT_SUITE_FILES: tuple[str, ...] = (
    "kv_cache_ops.mlir",
    "kv_cache_optimization.mlir",
    "mvp_single_layer_pipeline.mlir",
    "decoder_workload_buckets.mlir",
)


@dataclass
class LitRunResult:
    """Outcome for one lit file."""

    path: str
    status: str  # passed | failed | skipped
    runs: List[Dict[str, str]] = field(default_factory=list)
    reason: str = ""


def default_lit_dir(root: Optional[Path] = None) -> Path:
    root = root or Path(__file__).resolve().parents[3]
    return root / "test/Dialect/LLM"


def list_lit_files(lit_dir: Optional[Path] = None) -> List[Path]:
    lit_dir = lit_dir or default_lit_dir()
    return [lit_dir / name for name in LIT_SUITE_FILES]


def parse_run_lines(mlir_text: str) -> List[str]:
    return [match.group(1).strip() for match in RUN_LINE_RE.finditer(mlir_text)]


def _mlir_opt_command(run_line: str, mlir_path: Path) -> Optional[List[str]]:
    """Extract the mlir-opt / llmir-opt invocation before any pipe."""
    segment = run_line.split("|", 1)[0].strip()
    if "mlir-opt" not in segment and "llmir-opt" not in segment:
        return None
    parts = segment.split()
    cmd: List[str] = []
    for part in parts:
        if part == "%s":
            cmd.append(str(mlir_path))
        elif part.startswith("%"):
            continue
        else:
            cmd.append(part)
    executable = find_mlir_opt()
    if not executable:
        return None
    if cmd and cmd[0] in ("mlir-opt", "llmir-opt"):
        cmd[0] = executable
    elif executable:
        cmd.insert(0, executable)
    return cmd


def run_lit_file(mlir_path: Path) -> LitRunResult:
    """Run mlir-opt stages for one lit file; skip when opt is missing."""
    text = mlir_path.read_text(encoding="utf-8")
    runs = parse_run_lines(text)
    if not runs:
        return LitRunResult(str(mlir_path), "failed", reason="no RUN lines")

    executable = find_mlir_opt()
    if not executable:
        return LitRunResult(
            str(mlir_path),
            "skipped",
            runs=[{"line": line, "status": "skipped"} for line in runs],
            reason="mlir-opt / llmir-opt not on PATH",
        )

    row_logs: List[Dict[str, str]] = []
    for line in runs:
        cmd = _mlir_opt_command(line, mlir_path)
        if cmd is None:
            row_logs.append(
                {"line": line, "status": "skipped", "detail": "no mlir-opt stage"}
            )
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            row_logs.append(
                {
                    "line": line,
                    "status": "failed",
                    "detail": (proc.stderr or proc.stdout or "")[:500],
                }
            )
            return LitRunResult(
                str(mlir_path), "failed", runs=row_logs, reason="mlir-opt failed"
            )
        row_logs.append({"line": line, "status": "passed"})

    return LitRunResult(str(mlir_path), "passed", runs=row_logs)


def run_lit_suite(
    lit_dir: Optional[Path] = None,
    *,
    files: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Run all cataloged lit files; return summary dict."""
    lit_dir = lit_dir or default_lit_dir()
    names = list(files) if files is not None else list(LIT_SUITE_FILES)
    results = [run_lit_file(lit_dir / name) for name in names]
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    return {
        "status": (
            "failed" if failed else ("skipped" if skipped == len(results) else "passed")
        ),
        "mlir_opt": find_mlir_opt(),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "files": [
            {
                "path": r.path,
                "status": r.status,
                "reason": r.reason,
                "runs": r.runs,
            }
            for r in results
        ],
    }
