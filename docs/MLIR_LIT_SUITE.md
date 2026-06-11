# MLIR Lit Suite (Tier-A catalog)

Python-discoverable MLIR lit tests under `test/Dialect/LLM/`. FileCheck stages run under LLVM lit when built; the Python suite runs **mlir-opt stages only** for CI-friendly smoke when `mlir-opt` / `llmir-opt` is on `PATH`.

## Catalog

| File | Focus |
|------|--------|
| `kv_cache_ops.mlir` | Dialect op syntax |
| `kv_cache_optimization.mlir` | E1 block-size rewrite |
| `mvp_single_layer_pipeline.mlir` | Optimize + lower hot path |
| `decoder_workload_buckets.mlir` | S1/S2/S3 trace shapes (Loop R10) |

`optimization_passes.mlir` is aspirational (`llmir-opt` pipeline demos) and **not** in the Tier-A catalog.

## Commands

```bash
pytest tests/test_mlir_lit_suite.py -q
python3 scripts/verify_mlir_lit_suite.py
```

Without MLIR build: tests pass with `status=skipped` (honest). With `mlir-opt`:

```bash
export PATH=/path/to/llvm-build/bin:$PATH
pytest tests/test_mlir_lit_suite.py -q
```

## Output

`IEEE-conference/benchmarks/mlir_lit_suite_status.json`
