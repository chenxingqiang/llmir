#!/usr/bin/env python3
"""Block-size optimization — Nature style (E1 aligned illustrative sweep)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nature_style import NATURE_COLORS, apply_nature_style, despine, panel_label, save_figure

HERE = Path(__file__).resolve().parent

# Illustrative sweep (ShareGPT-shaped workload; replace via E1 JSON when available)
BLOCK_SIZES = np.array([16, 32, 64, 128, 256])
THROUGHPUT = np.array([43479, 43318, 41670, 42181, 48407])
FRAGMENTATION = np.array([15.0, 12.0, 8.5, 6.2, 4.8])  # lower is better
OPTIMAL_IDX = int(np.argmax(THROUGHPUT))


def main() -> None:
    apply_nature_style(base_size=8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    width = 28

    colors = ["#D9D9D9"] * len(BLOCK_SIZES)
    colors[OPTIMAL_IDX] = NATURE_COLORS[2]
    ax1.bar(BLOCK_SIZES, THROUGHPUT / 1000, width=width, color=colors, edgecolor="none", zorder=2)
    ax1.plot(BLOCK_SIZES, THROUGHPUT / 1000, color=NATURE_COLORS[3], marker="o", linewidth=1.0, zorder=3)
    ax1.set_xlabel("Block size (tokens)")
    ax1.set_ylabel("Throughput (×10³ tok s$^{-1}$)")
    ax1.set_xticks(BLOCK_SIZES)
    despine(ax1)
    panel_label(ax1, "a")

    ax2.plot(
        BLOCK_SIZES,
        FRAGMENTATION,
        color=NATURE_COLORS[0],
        marker="s",
        linewidth=1.1,
        markersize=4,
    )
    ax2.axvline(BLOCK_SIZES[OPTIMAL_IDX], color="#AAAAAA", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Block size (tokens)")
    ax2.set_ylabel("Fragmentation (%)")
    ax2.set_xticks(BLOCK_SIZES)
    despine(ax2)
    panel_label(ax2, "b")

    fig.text(
        0.5,
        0.02,
        "ILLUSTRATIVE — KV-append microbench lineage (see benchmark_summary.txt); not A100 LLaMA e2e.",
        ha="center",
        fontsize=6.5,
        color="#666666",
    )
    fig.subplots_adjust(bottom=0.18, wspace=0.38)

    save_figure(fig, "block_size_optimization_nature", out_dir=HERE)
    print(f"Wrote {HERE / 'block_size_optimization_nature.pdf'}")


if __name__ == "__main__":
    main()
