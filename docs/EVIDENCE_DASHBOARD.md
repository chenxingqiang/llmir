# LLMIR Evidence Dashboard

> Auto-generated. Regenerate: `python3 scripts/generate_evidence_dashboard.py`

## CI status

[![A-class walkthrough](https://github.com/chenxingqiang/llmir/actions/workflows/a-class-walkthrough.yml/badge.svg)](https://github.com/chenxingqiang/llmir/actions/workflows/a-class-walkthrough.yml)
[![Python package](https://github.com/chenxingqiang/llmir/actions/workflows/python-package.yml/badge.svg)](https://github.com/chenxingqiang/llmir/actions/workflows/python-package.yml)

## Summary

| Signal | Value |
|--------|-------|
| Package (local) | `0.2.2` |
| PyPI latest | `n/a` (unavailable) |
| M6 artifact bundle | **pass** (13 entries) |
| E8 empirical GPU | `skipped` (expected on CPU CI) |
| MLIR lit suite | `skipped` (needs mlir-opt on PATH) |
| Native build prereqs | `missing llvm` |

## Lab snapshot (optional)

| Check | Status |
|-------|--------|
| mlir lit | `skipped` |
| E8 GPU | `skipped` |
| PyPI publish | `missing` |

Regenerate: `bash scripts/lab_smoke_all.sh`

## Artifact rows

| ID | Experiment | Status |
|----|------------|--------|
| `e1_mlir_snippet` | E1 | ok |
| `e2_prefix_sim` | E2 | ok |
| `e4_compositional` | E4 | ok |
| `e5_ablation` | E5 | ok |
| `e6_backend_parity` | E6 | ok |
| `m5_lowered_hot_path` | M5 | ok |
| `paper_results` | paper | ok |
| `e8_empirical_gpu` | E8 | ok |
| `e4_compositional_buckets` | E4 | ok |
| `e5_ablation_buckets` | E5 | ok |
| `walkthrough_summary` | M6 | ok |
| `pypi_release_status` | release | ok |
| `lab_status_summary` | lab | ok |

## Commands

```bash
bash scripts/walkthrough_a_class.sh
python3 scripts/walkthrough_summary.py
bash scripts/lab_smoke_all.sh
bash scripts/reproduce_paper.sh
```

See also: `docs/WALKTHROUGH.md`, `docs/LOOP_MILESTONE_STATUS.md`, `docs/LAB_RUNBOOK.md`.
