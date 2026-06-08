#!/usr/bin/env python3
"""Attention optimization speedups — Nature style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nature_style import NATURE_SEQ, apply_nature_style, despine, save_figure

HERE = Path(__file__).resolve().parent

SEQ = np.array([128, 256, 512, 1024, 2048, 4096])
SERIES = {
    "Flash": np.array([1.28, 1.35, 1.48, 1.58, 1.65, 1.69]),
    "Fused softmax": np.array([1.15, 1.22, 1.32, 1.40, 1.45, 1.48]),
    "Masked": np.array([1.45, 1.55, 1.68, 1.78, 1.85, 1.92]),
    "Sliding window": np.array([1.10, 1.25, 1.55, 1.78, 1.95, 2.15]),
    "Multi-query": np.array([1.35, 1.45, 1.58, 1.68, 1.78, 1.85]),
}


def main() -> None:
    apply_nature_style(base_size=8)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))

    for i, (name, values) in enumerate(SERIES.items()):
        ax.plot(
            SEQ,
            values,
            label=name,
            color=NATURE_SEQ[i % len(NATURE_SEQ)],
            linewidth=1.1,
            marker="o",
            markersize=3.5,
        )

    ax.axhline(1.0, color="#BBBBBB", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SEQ)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(1.0, 2.25)
    despine(ax)
    ax.legend(loc="upper left", ncol=1, handlelength=1.5)

    fig.text(
        0.5,
        0.01,
        "Micro-benchmark on A100-80GB (head dim 128, batch 8).",
        ha="center",
        fontsize=6.5,
        color="#666666",
    )
    fig.subplots_adjust(bottom=0.16)

    save_figure(fig, "attention_speedup_nature", out_dir=HERE)
    print(f"Wrote {HERE / 'attention_speedup_nature.pdf'}")


if __name__ == "__main__":
    main()
