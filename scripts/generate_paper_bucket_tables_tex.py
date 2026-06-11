#!/usr/bin/env python3
"""Generate LaTeX tables for E4/E5 multi-bucket traces from committed JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E4_JSON = ROOT / "IEEE-conference/benchmarks/e4_compositional_buckets.json"
E5_JSON = ROOT / "IEEE-conference/benchmarks/e5_ablation_buckets.json"
OUT_TEX = ROOT / "IEEE-conference/generated/e4_e5_bucket_tables.tex"


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def _speedup(x: float) -> str:
    return f"{x:.2f}$\\times$"


def build_tex(e4: dict, e5: dict) -> str:
    lines = [
        "% Auto-generated from e4_compositional_buckets.json and e5_ablation_buckets.json",
        "% Regenerate: python3 scripts/generate_paper_bucket_tables_tex.py",
        "",
        "\\begin{table}[htbp]",
        "\\caption{E4 compositional proxies across decoder workload buckets (S1--S3)\\textsuperscript{$\\ddagger$}}",
        "\\begin{center}",
        "\\small",
        "\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Bucket} & $\\mathbf{L_s}$ & $\\mathbf{N}$ & $\\mathbf{L_u}$ & "
        "\\textbf{E1} & \\textbf{E2} & \\textbf{E3} & \\textbf{Sim} & \\textbf{Ideal} \\\\",
        "\\hline",
    ]
    for row in e4["buckets"]:
        t = row["analysis"]["trace"]
        c = row["analysis"]["composite"]["compile_time_levers"]
        mc = row["analysis"]["measured_comparison"]
        lines.append(
            f"{row['bucket_id']} & {t['shared_prefix_tokens']} & {t['num_requests']} & "
            f"{t['suffix_tokens']} & {_pct(c['e1_block_size_reduction'])} & "
            f"{_pct(c['e2_prefill_token_reduction'])} & {_pct(c['e3_host_copy_elimination'])} & "
            f"{_speedup(mc['measured_kv_sim_speedup'])} & {_speedup(mc['ideal_kv_speedup_model'])} \\\\"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\label{tab:e4_buckets}",
            "\\end{center}",
            "\\textsuperscript{$\\ddagger$}\\footnotesize Source: \\texttt{e4\\_compositional\\_buckets.json}; "
            "all rows satisfy \\texttt{measured\\_within\\_ideal\\_bound}=true.",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "\\caption{E5 full-stack proxy ablation across decoder workload buckets (S1--S3)\\textsuperscript{$\\ddagger$}}",
            "\\begin{center}",
            "\\small",
            "\\begin{tabular}{|l|c|c|c|c|c|c|}",
            "\\hline",
            "\\textbf{Bucket} & $\\mathbf{L_s}$ & $\\mathbf{N}$ & $\\mathbf{L_u}$ & "
            "\\textbf{E1} & \\textbf{E2} & \\textbf{E3} \\\\",
            "\\hline",
        ]
    )
    for row in e5["buckets"]:
        t = row["ablation"]["trace"]
        full = next(c for c in row["ablation"]["configurations"] if c["name"] == "full")
        p = full["proxies"]
        lines.append(
            f"{row['bucket_id']} & {t['shared_prefix_tokens']} & {t['num_requests']} & "
            f"{t['suffix_tokens']} & {_pct(p['block_size_reduction_ratio'])} & "
            f"{_pct(p['prefill_reduction_ratio'])} & {_pct(p['host_copy_reduction_ratio'])} \\\\"
        )
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "\\label{tab:e5_buckets}",
            "\\end{center}",
            "\\textsuperscript{$\\ddagger$}\\footnotesize Source: \\texttt{e5\\_ablation\\_buckets.json}; "
            "isolated/cumulative rows omitted for space.",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    e4 = json.loads(E4_JSON.read_text(encoding="utf-8"))
    e5 = json.loads(E5_JSON.read_text(encoding="utf-8"))
    tex = build_tex(e4, e5)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
