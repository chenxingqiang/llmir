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
| 6 | MLIR lit | Catalog tests; `mlir-opt` runs when on PATH |
| 7 | E8 (optional) | B-class GPU bench; `skipped` without CUDA is OK |
| 8 | M6 | `artifact_manifest.json` + committed JSON |

## Outputs to inspect

| File | Meaning |
|------|---------|
| `walkthrough_summary.json` | One-page reviewer snapshot (m6/e8/lit) |
| `artifact_bundle_status.json` | M6 verify summary |
| `e4_compositional_buckets.json` | S1/S2/S3 compositional proxies |
| `e5_ablation_buckets.json` | S1/S2/S3 ablation matrix |
| `e8_empirical_gpu.json` | B-class; `status=skipped` on CPU CI |
| `mlir_lit_suite_status.json` | Lit runner; `skipped` without mlir-opt |

## Full reproduce vs walkthrough

| Command | Scope | Time |
|---------|-------|------|
| `walkthrough_a_class.sh` | Verify committed artifacts + pytest gates | ~5 min CPU |
| `reproduce_paper.sh` | Regen JSON, figures, multi-bucket TeX, full verify | longer |

## Demo recording (optional)

Record a terminal session for reviewers:

```bash
bash scripts/walkthrough_a_class.sh
# ends with walkthrough_summary.json
```

Expected tail: `m6_all_pass: True`, `e8_status: skipped` on CPU CI (honest).

## GPU / MLIR lab follow-ups

```bash
# B-class E8 (needs CUDA + llmir[full])
python3 scripts/e8_empirical_gpu_bench.py

# MLIR lit smoke (needs mlir-opt on PATH)
export PATH=/path/to/llvm-build/bin:$PATH
python3 scripts/verify_mlir_lit_suite.py
```
