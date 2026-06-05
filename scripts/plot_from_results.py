#!/usr/bin/env python3
"""Plot benchmark JSON produced by llmir-benchmark or cpu_inference_compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots from reproducible benchmark JSON (no hard-coded throughput)."
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
        help="Output image path (requires matplotlib)",
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

    engines = [r.get("engine", r.get("backend", "?")) for r in rows]
    throughputs = [float(r.get("throughput_tokens_s", r.get("throughput", 0))) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(engines, throughputs)
    ax.set_ylabel("tokens/s")
    ax.set_title(f"Benchmark: {path.name}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
