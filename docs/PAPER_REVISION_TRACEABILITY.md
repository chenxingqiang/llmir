# Paper revision traceability (ICCD 2025 → code)

Maps reviewer-facing claims in `IEEE-conference/REVISION_NOTES.md` to **verifiable** artifacts in this repository.

## Paper experiment naming (E1–E3)

| Paper ID | Full name | Legacy CLI / tests | Doc |
|----------|-----------|-------------------|-----|
| **E1** | Compile-Time Pass Verification | `llmir-compile --mvp-a-e2e`, `tests/test_mvp_a_e2e.py` | `docs/E1_COMPILE_PASS_VERIFICATION.md` |
| **E2** | Prefix-Aware Serving Evaluation | `llmir-benchmark --sharegpt-prefix-bench`, `scripts/sharegpt_prefix_bench.py` | `docs/E2_PREFIX_SERVING_EVAL.md` |
| **E3** | GPU-Resident KV Integration | `llmir-benchmark --mvp-c-bench`, `tests/test_mvp_c_e2e.py` | `docs/E3_GPU_KV_INTEGRATION.md` |

Repository code and CI retain legacy `mvp-*` flag names; the paper uses **E1–E3** only.

| Revision item | Verification | Status |
|---------------|--------------|--------|
| §4 Algorithm 1 block size optimization | `lib/Dialect/LLM/Transforms/BlockSizeAnalysis.cpp`, `src/llmir/compiler/block_size.py` | Implemented |
| `llm-optimize-kv-cache` applies block size | `KVCacheOptimization.cpp` calls `applyBlockSizeOptimizationToFunc` | Implemented |
| Lit: block size rewrite | `test/Dialect/LLM/kv_cache_optimization.mlir`, `mvp_single_layer_pipeline.mlir` | Implemented |
| §3.1 model → IR → kernel (single layer) | `llmir-compile --mvp-a-e2e`, `src/llmir/compiler/mvp_pipeline.py` | **E1** |
| Lower to runtime calls | `-llm-lower-kv-cache-ops` → `@mlir_llm_*` | Implemented (needs `llmir-opt`) |
| Reference correctness | `tests/test_mvp_a_e2e.py`, `tests/test_compile_e2e.py` | CI (Python) |
| §5 ShareGPT prefix / TTFT workload | `scripts/sharegpt_prefix_bench.py`, `llmir-benchmark --sharegpt-prefix-bench` | **E2** (sim + llmir_paged E2E) |
| GPU KV without CPU NumPy round-trip | `TorchGpuPagedKVCache`, `PagedKVDecoder` GPU path, `llmir-benchmark --mvp-c-bench` | **E3** |
| Native CUDA KV kernels | `libMLIRLLMRuntime` + `cuda_probe`, `LLMIR_KV_BACKEND=native` | **E3** (optional build) |
| §5 evaluation scope | Main text: compiler verification + serving proxies only | **No operator-level claims** |
| Operator / FlashAttention speedup | Appendix `app:future_ops` only | **Not LLMIR kernels** — `benchmark/attention/` toys |
| Main-text throughput | Table measured_harness only | **gpt2 CPU** + **cited Qwen** |
| Appendix scale-out | `app:projected` design targets | **Illustrative** |
| Prefix TTFT Fig (2048) | `sharegpt_2048_sim.json` | **KV sim measured** |

## Quick commands

```bash
# Python-only E1 (no MLIR build required)
pytest tests/test_mvp_a_e2e.py -m "not network" -q

# E3 torch GPU KV path
pytest tests/test_torch_gpu_kv_cache.py tests/test_mvp_c_e2e.py -m "not network" -q
llmir-benchmark --mvp-c-bench --model gpt2 -o mvp_c.json

# Paper measured JSON + figures
python3 scripts/paper_benchmark_collect.py --model gpt2
python3 IEEE-conference/figures/create_measured_figures_nature.py

# Full path when llmir-opt is on PATH
llmir-compile --mvp-a-e2e --run-opt --run-reference --compare-torch \
  --seq-len 8 --mvp-json /tmp/mvp_a.json -o /tmp/mvp_a.mlir

# MLIR lit (LLVM/MLIR build tree)
mlir-opt test/Dialect/LLM/mvp_single_layer_pipeline.mlir -llm-optimize-kv-cache
```

## Paper figures (Nature style)

Regenerate measured: `python3 IEEE-conference/figures/generate_all_nature_figures.py`

Regenerate projected/illustrative: `python3 IEEE-conference/figures/generate_projected_figures.py`

| Figure | Verified? | Source |
|--------|-----------|--------|
| `e1_e3_evaluation_nature` | **Yes** (panels a–c) | E1/E2/E3 CI + docs |
| `prefix_cache_nature` | **Partial** | `sharegpt_prefix_bench` |
| `block_size_optimization_nature` | **Partial** | Algorithm 1 + illustrative sweep |
| `multi_model_comparison_nature` | **No** (projected) | Appendix A design targets |
| `attention_speedup_nature` | **No** (future work) | Appendix `app:future_ops`; not LLMIR lowered kernels |

## Honesty notes

- Main text §5: **E1–E3 + serving proxies only**; no operator-level kernel claims.
- Throughput heatmap / scale-out tables: **projected** in Appendix `app:projected`.
- Attention figure: **future operator work** in Appendix `app:future_ops`; not evidence for LLMIR codegen today.
