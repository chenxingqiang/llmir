# Paper revision traceability (ICCD 2025 → code)

Maps reviewer-facing claims in `IEEE-conference/REVISION_NOTES.md` to **verifiable** artifacts in this repository.

| Revision item | Verification | Status |
|---------------|--------------|--------|
| §4 Algorithm 1 block size optimization | `lib/Dialect/LLM/Transforms/BlockSizeAnalysis.cpp`, `src/llmir/compiler/block_size.py` | Implemented |
| `llm-optimize-kv-cache` applies block size | `KVCacheOptimization.cpp` calls `applyBlockSizeOptimizationToFunc` | Implemented |
| Lit: block size rewrite | `test/Dialect/LLM/kv_cache_optimization.mlir`, `mvp_single_layer_pipeline.mlir` | Implemented |
| §3.1 model → IR → kernel (single layer) | `llmir-compile --mvp-a-e2e`, `src/llmir/compiler/mvp_pipeline.py` | MVP-A |
| Lower to runtime calls | `-llm-lower-kv-cache-ops` → `@mlir_llm_*` | Implemented (needs `llmir-opt`) |
| Reference correctness | `tests/test_mvp_a_e2e.py`, `tests/test_compile_e2e.py` | CI (Python) |
| §5 ShareGPT throughput vs vLLM | — | **Not in MVP-A** (see MVP-B roadmap) |
| Table III PPL / MMLU | — | Planned |
| Multi-model Table II | — | Planned (requires GPU harness) |

## Quick commands

```bash
# Python-only MVP-A (no MLIR build required)
pytest tests/test_mvp_a_e2e.py -m "not network" -q

# Full path when llmir-opt is on PATH
llmir-compile --mvp-a-e2e --run-opt --run-reference --compare-torch \
  --seq-len 8 --mvp-json /tmp/mvp_a.json -o /tmp/mvp_a.mlir

# MLIR lit (LLVM/MLIR build tree)
mlir-opt test/Dialect/LLM/mvp_single_layer_pipeline.mlir -llm-optimize-kv-cache
```

## Honesty notes

- Throughput figures in `IEEE-conference/figures/paper-only/` are **not** produced by this pipeline.
- MVP-A proves **compile-time block sizing + single-layer IR lowering + numeric reference**; it does not claim vLLM-scale serving wins.
