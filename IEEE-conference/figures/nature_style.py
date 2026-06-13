"""
Nature-journal-inspired matplotlib style for LLMIR paper figures.

Guidelines adapted from Nature figure specifications:
- Single-column width 89 mm; double-column 183 mm (at 300 dpi export)
- Sans-serif typography (Arial / Helvetica fallback)
- Colour-blind-friendly palette (ggsci::nature)
- Minimal chartjunk; despine top/right
- Panel labels a, b, c in bold sans-serif
- Thin axes (~0.5–0.8 pt), white background, editable PDF fonts
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
SINGLE_COL_MM = 89.0
DOUBLE_COL_MM = 183.0

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

NATURE_MUTED: Tuple[str, ...] = (
    "#D9D9D9",
    "#E8E8E8",
    "#C8C8C8",
)


def figsize_mm(width_mm: float, height_mm: float) -> Tuple[float, float]:
    """Convert Nature layout width/height in millimetres to matplotlib inches."""
    return (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)


def apply_nature_style(*, base_size: float = 7.0) -> None:
    """Apply global rcParams for Nature-style figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": base_size - 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "patch.linewidth": 0.0,
            "mathtext.fontset": "dejavusans",
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


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.12,
    fontsize: float = 8.0,
) -> None:
    """Add bold panel label (a, b, c) outside axes — Nature convention."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        color="#222222",
    )


def source_footnote(
    fig: plt.Figure,
    text: str,
    *,
    y: float = 0.02,
    fontsize: float = 6.0,
    color: str = "#666666",
) -> None:
    """Centered data-source or honesty footnote below panels."""
    fig.text(0.5, y, text, ha="center", fontsize=fontsize, color=color)


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
