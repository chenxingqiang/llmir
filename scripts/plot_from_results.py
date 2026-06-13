#!/usr/bin/env python3
"""Plot benchmark JSON produced by llmir-benchmark or cpu_inference_compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "IEEE-conference" / "figures"
sys.path.insert(0, str(FIGURES))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Nature-style plots from reproducible benchmark JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="JSON file (list of rows with engine, throughput_tokens_s, ...)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_plot.png",
        help="Output image path (PDF if .pdf extension)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        print(f"Missing input: {path}", file=sys.stderr)
        return 1

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        print("Empty benchmark JSON", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required for plotting. Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return 1

    from nature_style import NATURE_COLORS, SINGLE_COL_MM, apply_nature_style, despine, figsize_mm, source_footnote

    engines = [r.get("engine", r.get("backend", "?")) for r in rows]
    throughputs = [float(r.get("throughput_tokens_s", r.get("throughput", 0))) for r in rows]
    colors = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(engines))]

    apply_nature_style(base_size=7)
    fig, ax = plt.subplots(figsize=figsize_mm(SINGLE_COL_MM, 62))
    ax.bar(engines, throughputs, color=colors, width=0.55, edgecolor="none")
    ax.set_ylabel("Throughput (tok s$^{-1}$)")
    ax.set_title(f"Benchmark: {path.name}", fontsize=7)
    despine(ax)
    source_footnote(fig, f"Source: {path.name}", y=0.02)
    fig.subplots_adjust(bottom=0.14)

    out = Path(args.output)
    if out.suffix.lower() == ".pdf":
        fig.savefig(out, facecolor="white", edgecolor="none")
    else:
        fig.savefig(out, dpi=300, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
