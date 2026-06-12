# A-Class Walkthrough (CPU, ~5 min)

Reviewer-oriented path through **E1–E6 + M5/M6** without a full artifact regen. For one-shot reproduce (figures + JSON rewrite), use `bash scripts/reproduce_paper.sh`.

## Quick start

```bash
bash scripts/walkthrough_a_class.sh
```

## What each step checks

| Step | Evidence | Pass criterion |
|------|----------|----------------|
| 1 | E1 | `test_mvp_a_e2e.py` |
| 2 | E2 + buckets | ShareGPT sim + S1/S2/S3 JSON verify |
| 3 | E3 | MVP-C / torch KV CPU tests |
| 4 | E4/E5 | Compositional + ablation + multi-bucket |
| 5 | E6/M5 | Backend parity + lowered hot path |
| 6 | MLIR lit | `mlir_lit_smoke.sh`; `skipped` without mlir-opt OK |
| 7 | PyPI | `verify_pypi_release.py`; `pending` until published OK |
| 8 | E8 (optional) | `e8_lab_smoke.sh`; `skipped` without CUDA OK |
| 9 | M6 | `artifact_manifest.json` + committed JSON |

## Outputs to inspect

| File | Meaning |
|------|---------|
| `walkthrough_summary.json` | One-page reviewer snapshot |
| `lab_status_summary.json` | Optional lab rollup (mlir/e8/pypi/native) |
| `docs/EVIDENCE_DASHBOARD.md` | Human-readable dashboard (auto-generated) |
| `artifact_bundle_status.json` | M6 verify summary |
| `pypi_release_status.json` | Local vs PyPI version |
| `e8_empirical_gpu.json` | B-class; `status=skipped` on CPU CI |
| `mlir_lit_suite_status.json` | Lit runner; `skipped` without mlir-opt |

## Full reproduce vs walkthrough

| Command | Scope | Time |
|---------|-------|------|
| `walkthrough_a_class.sh` | Verify committed artifacts + pytest gates | ~5 min CPU |
| `reproduce_paper.sh` | Regen JSON, figures, multi-bucket TeX, lab tail | longer |
| `lab_smoke_all.sh` | Optional lab checks only | ~2 min |

## Demo recording (optional)

```bash
bash scripts/walkthrough_a_class.sh
```

Expected tail: `m6_all_pass: True`, `e8_status: skipped` on CPU CI (honest).

## Lab follow-ups

See [`LAB_RUNBOOK.md`](LAB_RUNBOOK.md) for mlir-opt build, E8 GPU strict lab, and PyPI republish workflows.
