#!/usr/bin/env python3
"""E2 ShareGPT-style prefix cache — Nature style."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES))
from nature_style import NATURE_COLORS, apply_nature_style, despine, panel_label, save_figure  # noqa: E402

HERE = Path(__file__).resolve().parent

# Illustrative: 32 requests, 128-token shared system prompt
REQUESTS = np.arange(1, 33)
COLD = np.linspace(1.0, 0.98, 32)
WARM = np.concatenate(
    [[1.0], np.linspace(0.35, 0.22, 31)]
)  # first req cold, then prefix hits


def main() -> None:
    apply_nature_style(base_size=8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax1.plot(REQUESTS, COLD, color=NATURE_COLORS[4], linewidth=1.1, label="No warm prefix")
    ax1.plot(REQUESTS, WARM, color=NATURE_COLORS[2], linewidth=1.1, label="After warm_prefix")
    ax1.set_xlabel("Request index")
    ax1.set_ylabel("Normalized end-to-end time")
    ax1.legend(loc="upper right")
    despine(ax1)
    panel_label(ax1, "a")

    cats = ["KV sim\nbaseline", "KV sim\nprefix", "llmir_paged\ncold", "llmir_paged\nwarm"]
    speedup = [1.0, 4.5, 1.0, 1.8]
    ax2.bar(cats, speedup, color=[NATURE_COLORS[4], NATURE_COLORS[1], NATURE_COLORS[4], NATURE_COLORS[2]], width=0.6, edgecolor="none")
    ax2.axhline(1.0, color="#CCCCCC", linewidth=0.8)
    ax2.set_ylabel("Speedup vs baseline")
    ax2.set_ylim(0, 5.2)
    despine(ax2)
    panel_label(ax2, "b")
    ax2.tick_params(axis="x", labelsize=6.5)

    fig.text(
        0.5,
        0.02,
        "E2 harness: sharegpt_prefix_bench.py / llmir-benchmark --sharegpt-prefix-bench.",
        ha="center",
        fontsize=6.5,
        color="#666666",
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.38)

    save_figure(fig, "prefix_cache_nature", out_dir=HERE)
    print(f"Wrote {HERE / 'prefix_cache_nature.pdf'}")


if __name__ == "__main__":
    main()
