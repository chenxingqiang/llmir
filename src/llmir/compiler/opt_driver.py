"""Invoke mlir-opt / llmir-opt on LLM dialect modules."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class OptResult:
    """Result of an optimizer invocation."""

    success: bool
    stdout: str
    stderr: str
    executable: str
    args: List[str]


def find_mlir_opt() -> Optional[str]:
    """Resolve mlir-opt or llmir-opt executable."""
    for env_key in ("LLMIR_OPT_EXECUTABLE", "MLIR_OPT_EXECUTABLE"):
        path = os.environ.get(env_key)
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for name in ("llmir-opt", "mlir-opt"):
        found = shutil.which(name)
        if found:
            return found
    return None


def run_mlir_opt(
    mlir_text: str,
    *,
    passes: Sequence[str] = ("-llm-lower-kv-cache-ops",),
    extra_args: Sequence[str] = (),
) -> OptResult:
    """
    Run mlir-opt on in-memory MLIR text.

    Returns ``success=False`` when the executable is missing (CI without MLIR build).
    """
    executable = find_mlir_opt()
    if not executable:
        return OptResult(
            success=False,
            stdout="",
            stderr="mlir-opt / llmir-opt not found",
            executable="",
            args=list(passes),
        )

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.mlir"
        in_path.write_text(mlir_text, encoding="utf-8")
        cmd = [executable, str(in_path), *passes, *extra_args]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        return OptResult(
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            executable=executable,
            args=cmd[1:],
        )
