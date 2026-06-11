"""Generated E4/E5 bucket tables stay in sync with JSON artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_paper_bucket_tables_tex.py"
OUT_TEX = ROOT / "IEEE-conference/generated/e4_e5_bucket_tables.tex"


def test_bucket_tables_tex_matches_json():
    before = OUT_TEX.read_text(encoding="utf-8") if OUT_TEX.is_file() else ""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    after = OUT_TEX.read_text(encoding="utf-8")
    assert "\\label{tab:e4_buckets}" in after
    assert "\\label{tab:e5_buckets}" in after
    assert "S1 & 128" in after
    assert "S2 & 2048" in after
    assert "S3 & 8192" in after
    if before:
        assert before == after, "Regenerate and commit IEEE-conference/generated/e4_e5_bucket_tables.tex"
