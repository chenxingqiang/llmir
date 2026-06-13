"""Nature-style paper figure helpers and regeneration smoke tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "IEEE-conference" / "figures"


def test_nature_style_mm_helpers():
    sys.path.insert(0, str(FIGURES))
    from nature_style import DOUBLE_COL_MM, SINGLE_COL_MM, figsize_mm

    w, h = figsize_mm(SINGLE_COL_MM, 60)
    assert 3.0 < w < 4.0
    assert 2.0 < h < 3.0
    w2, _ = figsize_mm(DOUBLE_COL_MM, 60)
    assert w2 > w


def test_paper_uses_nature_figure_paths():
    tex = (ROOT / "IEEE-conference" / "LLMIR-paper-ICCD2025-revised.tex").read_text(encoding="utf-8")
    for stem in (
        "llmir_architecture_nature",
        "e1_e3_evaluation_nature",
        "prefix_ttft_nature",
        "gpt2_measured_latency_nature",
        "block_size_optimization_nature",
        "attention_speedup_nature",
        "multi_model_comparison_nature",
    ):
        assert stem in tex
    assert "_nature.pdf" in tex


def test_nature_figure_scripts_exist():
    for name in (
        "nature_style.py",
        "generate_all_nature_figures.py",
        "generate_projected_figures.py",
        "create_e1_e3_evaluation_nature.py",
        "create_measured_figures_nature.py",
    ):
        assert (FIGURES / name).is_file()


def test_e1_e3_loads_repository_json():
    script = FIGURES / "create_e1_e3_evaluation_nature.py"
    text = script.read_text(encoding="utf-8")
    assert "e4_compositional_buckets.json" in text
    assert "shared_prefix_decoder_2048_sim.json" in text


def test_regenerate_measured_nature_figures():
    proc = subprocess.run(
        [sys.executable, str(FIGURES / "generate_all_nature_figures.py")],
        cwd=str(FIGURES),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    for stem in (
        "llmir_architecture_nature.pdf",
        "e1_e3_evaluation_nature.pdf",
        "prefix_ttft_nature.pdf",
        "gpt2_measured_latency_nature.pdf",
    ):
        assert (FIGURES / stem).is_file(), stem
