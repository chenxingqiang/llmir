"""Generate reviewer evidence dashboard markdown from committed artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llmir.benchmark.artifact_bundle import verify_artifact_bundle
from llmir.benchmark.walkthrough_summary import build_walkthrough_summary


def build_evidence_dashboard_markdown(root: Optional[Path] = None) -> str:
    root = Path(__file__).resolve().parents[3] if root is None else root
    summary = build_walkthrough_summary(root)
    m6 = verify_artifact_bundle(root, check_figures=False)

    m6_icon = "pass" if summary["m6_all_pass"] else "fail"
    e8_note = (
        "expected on CPU CI"
        if summary["e8_status"] == "skipped"
        else "B-class empirical"
    )
    lit_note = (
        "needs mlir-opt on PATH"
        if summary["mlir_lit_suite_status"] == "skipped"
        else summary["mlir_lit_suite_status"]
    )

    lines = [
        "# LLMIR Evidence Dashboard",
        "",
        "> Auto-generated. Regenerate: `python3 scripts/generate_evidence_dashboard.py`",
        "",
        "## CI status",
        "",
        "[![A-class walkthrough](https://github.com/chenxingqiang/llmir/actions/workflows/a-class-walkthrough.yml/badge.svg)]"
        "(https://github.com/chenxingqiang/llmir/actions/workflows/a-class-walkthrough.yml)",
        "",
        "## Summary",
        "",
        "| Signal | Value |",
        "|--------|-------|",
        f"| M6 artifact bundle | **{m6_icon}** ({summary['artifact_count']} entries) |",
        f"| E8 empirical GPU | `{summary['e8_status']}` ({e8_note}) |",
        f"| MLIR lit suite | `{summary['mlir_lit_suite_status']}` ({lit_note}) |",
        "",
        "## Artifact rows",
        "",
        "| ID | Experiment | Status |",
        "|----|------------|--------|",
    ]
    for row in m6.artifacts:
        status = "ok" if row["ok"] else f"FAIL: {row.get('message', '')}"
        lines.append(f"| `{row['id']}` | {row.get('experiment', '')} | {status} |")

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            summary["walkthrough_command"],
            "python3 scripts/walkthrough_summary.py",
            summary["reproduce_command"],
            "```",
            "",
            "See also: `docs/WALKTHROUGH.md`, `docs/LOOP_MILESTONE_STATUS.md`.",
            "",
        ]
    )
    return "\n".join(lines)
