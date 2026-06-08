# Nature-style figures for LLMIR paper

## Generate all figures

```bash
cd IEEE-conference/figures
python3 generate_all_nature_figures.py
```

Requires `matplotlib`. Outputs PDF + PNG (300 dpi).

## Figure list

| File | Script | Paper label |
|------|--------|-------------|
| `llmir_architecture_nature.pdf` | `create_architecture_diagram_nature.py` | Fig. architecture |
| `mvp_evaluation_nature.pdf` | `create_mvp_evaluation_nature.py` | Fig. mvp_eval |
| `block_size_optimization_nature.pdf` | `create_block_size_chart_nature.py` | Fig. block_optimization |
| `attention_speedup_nature.pdf` | `create_attention_speedup_nature.py` | Fig. attention_speedup |
| `paper-only/multi_model_comparison_nature.pdf` | `paper-only/create_multi_model_comparison_nature.py` | Fig. multi_model |
| `paper-only/prefix_cache_nature.pdf` | `paper-only/create_prefix_cache_nature.py` | Fig. prefix_cache |

## Style module

Shared theme: `nature_style.py` (ggsci Nature palette, Arial, despine, panel labels a/b/c).

## Data honesty

- **MVP panels** map to CI-verified harnesses (see `docs/PAPER_REVISION_TRACEABILITY.md`).
- **Table II / heatmap / block sweep** use illustrative targets until `llmir-benchmark` JSON is wired via `scripts/plot_from_results.py`.
