#!/usr/bin/env python3
"""Multi-model throughput — Nature style heatmap (projected targets)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FIGURES))
from nature_style import SINGLE_COL_MM, apply_nature_style, figsize_mm, panel_label, save_figure, source_footnote  # noqa: E402

HERE = Path(__file__).resolve().parent

MODELS = [
    "LLaMA-2 7B",
    "LLaMA-2 13B",
    "LLaMA-2 70B",
    "Phi-3 3.8B",
    "Qwen-2 7B",
    "Qwen-2 14B",
    "Qwen-2 72B",
    "DeepSeek 16B",
]
FRAMEWORKS = ["LLMIR", "vLLM", "SGLang", "TRT-LLM", "MLC-LLM"]

# tokens/s (same as Table II — illustrative until GPU harness fills JSON)
DATA = np.array(
    [
        [89120, 72850, 64200, 85400, 68900],
        [58499, 47800, 42400, 55200, 44100],
        [12450, 10200, 9050, 11800, 9400],
        [142300, 116500, 103200, 135800, 109400],
        [86200, 70400, 62100, 82100, 66500],
        [48600, 39700, 35200, 46100, 37200],
        [11200, 9150, 8100, 10650, 8450],
        [52400, 42800, 37900, 49600, 40100],
    ],
    dtype=float,
)


def main() -> None:
    apply_nature_style(base_size=7)
    fig, ax = plt.subplots(figsize=figsize_mm(SINGLE_COL_MM, 78))

    # Normalize per model row for colour (shows relative ranking)
    row_max = DATA.max(axis=1, keepdims=True)
    norm = DATA / row_max

    im = ax.imshow(norm, aspect="auto", cmap="YlGnBu", vmin=0.85, vmax=1.0)
    ax.set_xticks(range(len(FRAMEWORKS)))
    ax.set_xticklabels(FRAMEWORKS, rotation=35, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=6.5)
    ax.set_xlabel("Framework")
    # Annotate LLMIR column values (compact)
    for i in range(len(MODELS)):
        val = DATA[i, 0]
        ax.text(
            0,
            i,
            f"{val/1000:.0f}k",
            ha="center",
            va="center",
            fontsize=6,
            color="#1a1a1a",
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Norm. throughput", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.text(
        0.5,
        0.01,
        "Projected single-GPU targets (A100-80GB); replace via llmir-benchmark JSON.",
        ha="center",
        fontsize=6,
        color="#666666",
    )
    fig.subplots_adjust(bottom=0.22, left=0.28)

    save_figure(fig, "multi_model_comparison_nature", out_dir=HERE)
    print(f"Wrote {HERE / 'multi_model_comparison_nature.pdf'}")


if __name__ == "__main__":
    main()
