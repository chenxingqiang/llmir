# Nature-style figures for LLMIR paper

## Generate measured figures (default)

```bash
cd IEEE-conference/figures
python3 generate_all_nature_figures.py
```

Produces CI-verified / measured panels: architecture, E1–E3 evaluation, prefix TTFT, gpt2 latency.

## Generate projected / illustrative figures

```bash
python3 generate_projected_figures.py
```

Produces block-size sweep, attention microbench, multi-model heatmap, illustrative prefix curves.
**Not** CI-verified end-to-end LLM measurements.

Requires `matplotlib`. Outputs PDF + PNG (300 dpi).

## Layout (Nature journal widths)

| Layout | Width | Helper |
|--------|-------|--------|
| Single column | 89 mm | `figsize_mm(SINGLE_COL_MM, height_mm)` |
| Double column | 183 mm | `figsize_mm(DOUBLE_COL_MM, height_mm)` |

Shared theme: `nature_style.py` — ggsci Nature palette, Arial/Helvetica, despine, bold panel labels `a`/`b`/`c`, `source_footnote()` for data honesty.

Generic JSON plots: `python3 scripts/plot_from_results.py -i benchmarks.json -o out.pdf`

## Figure list

| File | Script | Status |
|------|--------|--------|
| `llmir_architecture_nature.pdf` | `create_architecture_diagram_nature.py` | Measured pipeline diagram |
| `e1_e3_evaluation_nature.pdf` | `create_e1_e3_evaluation_nature.py` | E1/E2 from repo JSON; E3 illustrative |
| `prefix_ttft_nature.pdf` | `create_measured_figures_nature.py` | Measured / sim JSON |
| `gpt2_measured_latency_nature.pdf` | `create_measured_figures_nature.py` | Measured |
| `block_size_optimization_nature.pdf` | `create_block_size_chart_nature.py` | **Projected** — KV microbench lineage |
| `attention_speedup_nature.pdf` | `create_attention_speedup_nature.py` | **Future work** — Appendix `app:future_ops` |
| `paper-only/multi_model_comparison_nature.pdf` | `paper-only/...` | **Projected** — Table II targets |
| `paper-only/prefix_cache_nature.pdf` | `paper-only/...` | **Projected** — hand-tuned curves |

## Data honesty

See `docs/PAPER_REVISION_TRACEABILITY.md`. E1/E2 panels read `e4_compositional_buckets.json` and `shared_prefix_decoder_2048_sim.json` when present.
