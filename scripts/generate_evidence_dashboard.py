#!/usr/bin/env python3
"""Write docs/EVIDENCE_DASHBOARD.md from committed artifact status."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.evidence_dashboard import build_evidence_dashboard_markdown  # noqa: E402

OUT = ROOT / "docs/EVIDENCE_DASHBOARD.md"


def main() -> int:
    md = build_evidence_dashboard_markdown(ROOT)
    OUT.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
