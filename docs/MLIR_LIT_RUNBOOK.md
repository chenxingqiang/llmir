# MLIR Lit Runbook (Tier-A catalog)

Runs the four Python-cataloged lit files under `test/Dialect/LLM/` when a
**repo-built** `mlir-opt` (with LLM dialect passes) is available. Stock distro
`mlir-opt` binaries do **not** register `-llm-optimize-kv-cache`.

## Catalog (M8)

| File | Passes exercised |
|------|------------------|
| `kv_cache_ops.mlir` | parse / dialect ops |
| `kv_cache_optimization.mlir` | `-llm-optimize-kv-cache` |
| `mvp_single_layer_pipeline.mlir` | optimize + `-llm-lower-kv-cache-ops` |
| `decoder_workload_buckets.mlir` | S1/S2/S3 block-size buckets |

## Build mlir-opt

Requires CMake, a C++17 compiler, and LLVM/MLIR dev packages or monorepo build.
See `docs/MLIR_NATIVE_BUILD.md` and `bash scripts/check_native_build_prereqs.sh`.

```bash
bash scripts/build_mlir_opt.sh
export PATH="${PWD}/build-native/bin:$PATH"
# or: export LLMIR_OPT_EXECUTABLE="${PWD}/build-native/bin/mlir-opt"
```

Reuses `BUILD_DIR` (default `build-native/`) with `scripts/build_native_runtime.sh`.

## Smoke test

```bash
bash scripts/mlir_lit_smoke.sh
# writes IEEE-conference/benchmarks/mlir_lit_suite_status.json
```

Expected when opt is present: `status: passed`, four files green.

## CI note

A-class walkthrough and CPU CI **skip** lit when opt is absent (`status: skipped`).
`mlir_lit_smoke.sh` always writes `mlir_lit_suite_status.json` and exits 0 when
skipped (exit 1 only on `failed`).

**GitHub Actions**: workflow **MLIR lit lab (optional)** (`mlir-lit-lab.yml`)
accepts optional `mlir_opt_executable` and `require_passed=true` for strict lab closure.

```bash
bash scripts/mlir_lit_preflight.sh --strict /path/to/build-native/bin/mlir-opt
bash scripts/mlir_lit_smoke.sh
python3 scripts/verify_mlir_lit_suite.py --require-passed
```

Full lit closure is a **lab** step, not a PyPI wheel requirement.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `unknown pass 'llm-optimize-kv-cache'` | Use in-tree `mlir-opt`, not system package |
| `mlir-opt not found` | Run `build_mlir_opt.sh` and export `PATH` |
| FileCheck failures | Rebuild after changing `lib/Dialect/LLM/` passes |
