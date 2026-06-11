#!/usr/bin/env python3
"""Generate paper figures from IEEE-conference/benchmarks/paper_results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmarks"
sys.path.insert(0, str(HERE))
from nature_style import NATURE_COLORS, apply_nature_style, despine, panel_label, save_figure


def _load() -> dict:
    path = BENCH / "paper_results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_2048() -> dict:
    path = BENCH / "shared_prefix_decoder_2048_sim.json"
    if not path.exists():
        path = BENCH / "sharegpt_2048_sim.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def fig_gpt2_latency(data: dict) -> None:
    rows = {r["engine"]: r for r in data.get("inference_compare", [])}
    engines = ["hf", "llmir-paged"]
    labels = ["HF\ngenerate()", "LLMIR\nllmir-paged"]
    tps = [rows[e]["throughput_tokens_s"] for e in engines if e in rows]
    labels = [labels[i] for i, e in enumerate(engines) if e in rows]
    colors = [NATURE_COLORS[3], NATURE_COLORS[0][:7] + "CC"]

    apply_nature_style(base_size=8)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.bar(labels, tps, color=colors, width=0.55, edgecolor="none")
    ax.set_ylabel("Throughput (tok s$^{-1}$)")
    notes = data.get("notes", {})
    ax.set_title(
        f"gpt2 measured (batch=1, prompt={notes.get('prompt_tokens', '?')})",
        fontsize=8,
    )
    despine(ax)
    fig.text(
        0.5,
        0.01,
        "Source: scripts/paper_benchmark_collect.py → paper_results.json",
        ha="center",
        fontsize=6,
        color="#666666",
    )
    fig.subplots_adjust(bottom=0.18)
    save_figure(fig, "gpt2_measured_latency_nature", out_dir=HERE)


def fig_prefix_ttft(data: dict, sim2048: dict) -> None:
    apply_nature_style(base_size=8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # (a) KV simulation @ 2048 tokens
    if sim2048.get("results"):
        base = next(r for r in sim2048["results"] if "baseline" in r["scenario"])
        warm = next(r for r in sim2048["results"] if "prefix_cached" in r["scenario"])
        cats = ["No prefix", "Prefix cached"]
        lat = [base["avg_latency_ms"], warm["avg_latency_ms"]]
        ax1.bar(cats, lat, color=[NATURE_COLORS[4], NATURE_COLORS[2]], width=0.55, edgecolor="none")
        sp = warm.get("speedup_vs_baseline", base["avg_latency_ms"] / max(warm["avg_latency_ms"], 1e-9))
        ax1.set_ylabel("KV-layer latency (ms)")
        ax1.set_title(f"prefix={sim2048.get('config', {}).get('system_prompt_tokens', 2048)} tok", fontsize=8)
        ax1.text(0.95, 0.9, f"{sp:.1f}×", transform=ax1.transAxes, ha="right", fontsize=8, fontweight="bold")
    despine(ax1)
    panel_label(ax1, "a")

    # (b) llmir_paged prefill tokens saved (per request)
    sp = data.get("shared_prefix_decoder") or data.get("sharegpt_prefix", {})
    per = sp.get("per_request", {})
    if per.get("baseline") and per.get("warmed"):
        idx = np.arange(len(per["warmed"]))
        prefill_cold = [per["baseline"][i]["prefill_tokens_computed"] for i in range(len(per["warmed"]))]
        prefill_warm = [per["warmed"][i]["prefill_tokens_computed"] for i in range(len(per["warmed"]))]
        w = 0.35
        ax2.bar(idx - w / 2, prefill_cold, width=w, label="No prefix", color=NATURE_COLORS[4], edgecolor="none")
        ax2.bar(idx + w / 2, prefill_warm, width=w, label="warm_prefix", color=NATURE_COLORS[2], edgecolor="none")
        ax2.set_xlabel("Request index")
        ax2.set_ylabel("Prefill tokens computed")
        ax2.legend(loc="upper right", fontsize=7)
    despine(ax2)
    panel_label(ax2, "b")

    fig.text(
        0.5,
        0.02,
        "a: KV simulation (2048-token system prompt). b: gpt2 llmir_paged E2E prefill savings.",
        ha="center",
        fontsize=6.5,
        color="#666666",
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.38)
    save_figure(fig, "prefix_ttft_nature", out_dir=HERE)


def main() -> None:
    data = _load()
    sim2048 = _load_2048()
    fig_gpt2_latency(data)
    fig_prefix_ttft(data, sim2048)
    print("Measured figures written to", HERE)


if __name__ == "__main__":
    main()
