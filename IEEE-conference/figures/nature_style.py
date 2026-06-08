"""
Nature-journal-inspired matplotlib style for LLMIR paper figures.

Guidelines adapted from Nature figure specifications:
- Sans-serif typography (Arial / Helvetica fallback)
- Colour-blind-friendly palette (ggsci::nature)
- Minimal chartjunk; despine top/right
- Panel labels a, b, c in bold
- Thin axes (~0.8 pt), white background
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

# ggsci Nature palette (colour-blind friendly)
NATURE_COLORS: Tuple[str, ...] = (
    "#E64B35",  # red
    "#4DBBD5",  # cyan
    "#00A087",  # teal
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # gray-blue
    "#91D1C2",  # mint
    "#DC0000",  # crimson
    "#7E6148",  # brown
    "#B09C85",  # tan
)

NATURE_SEQ: Tuple[str, ...] = (
    "#3C5488",
    "#00A087",
    "#E64B35",
    "#4DBBD5",
    "#8491B4",
    "#F39B7F",
)


def apply_nature_style(*, base_size: float = 8.0) -> None:
    """Apply global rcParams for Nature-style figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size + 1,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": base_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def despine(ax: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    """Remove top/right spines (Nature default)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add bold panel label (a, b, c) outside axes."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color="#222222",
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Save PDF + PNG to ``out_dir`` (defaults to caller script directory)."""
    out = Path(out_dir) if out_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{stem}.pdf"
    png = out / f"{stem}.png"
    fig.savefig(pdf, facecolor="white", edgecolor="none")
    fig.savefig(png, facecolor="white", edgecolor="none")
    plt.close(fig)
    return pdf, png


def cycle_colors(n: int) -> List[str]:
    return [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(n)]
