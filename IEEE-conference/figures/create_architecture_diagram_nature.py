#!/usr/bin/env python3
"""LLMIR compilation pipeline — Nature style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nature_style import DOUBLE_COL_MM, NATURE_COLORS, NATURE_SEQ, apply_nature_style, figsize_mm, save_figure

HERE = Path(__file__).resolve().parent


def main() -> None:
    apply_nature_style(base_size=7)
    fig, ax = plt.subplots(figsize=figsize_mm(DOUBLE_COL_MM, 72))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("Model import", "PyTorch / ONNX", NATURE_SEQ[0]),
        ("LLM dialect opt.", "KV / attention IR", NATURE_SEQ[1]),
        ("Codegen", "CUDA / CPU", NATURE_SEQ[2]),
        ("Runtime", "Paged KV + batch", NATURE_SEQ[3]),
    ]
    x0, w, h, y = 0.3, 2.1, 1.05, 2.35
    gap = 0.35
    for i, (title, sub, color) in enumerate(stages):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor="none",
            alpha=0.18,
        )
        ax.add_patch(box)
        edge = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="none",
            edgecolor=color,
            linewidth=1.0,
        )
        ax.add_patch(edge)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7, color="#444444")
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.04, y + h / 2),
                    (x + w + gap - 0.04, y + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.9,
                    color="#555555",
                )
            )

    # E1–E3 traceability strip
    experiments = [
        ("E1", "Compile-time pass verification", NATURE_COLORS[1]),
        ("E2", "Prefix-aware serving eval.", NATURE_COLORS[2]),
        ("E3", "GPU-resident KV integration", NATURE_COLORS[0]),
    ]
    bw = 2.85
    for j, (tag, desc, c) in enumerate(experiments):
        bx = 0.35 + j * (bw + 0.25)
        ax.add_patch(
            FancyBboxPatch(
                (bx, 0.55),
                bw,
                0.85,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="white",
                edgecolor=c,
                linewidth=0.9,
            )
        )
        ax.text(bx + 0.12, 1.12, tag, fontsize=7.5, fontweight="bold", color=c, va="center")
        ax.text(bx + 0.55, 1.12, desc, fontsize=7, color="#333333", va="center")

    ax.text(
        5,
        3.72,
        "LLMIR end-to-end compilation and serving path",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#222222",
    )

    save_figure(fig, "llmir_architecture_nature", out_dir=HERE)
    print(f"Wrote {HERE / 'llmir_architecture_nature.pdf'}")


if __name__ == "__main__":
    main()
