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
| §5 ShareGPT prefix / TTFT workload | `scripts/sharegpt_prefix_bench.py`, `llmir-benchmark --sharegpt-prefix-bench` | **MVP-B** (sim + llmir_paged E2E) |
| GPU KV without CPU NumPy round-trip | `TorchGpuPagedKVCache`, `PagedKVDecoder` GPU path, `llmir-benchmark --mvp-c-bench` | **MVP-C** |
| Native CUDA KV kernels | `libMLIRLLMRuntime` + `cuda_probe`, `LLMIR_KV_BACKEND=native` | **MVP-C** (optional build) |
| §5 ShareGPT throughput vs vLLM | vLLM baseline in same harness | Planned (needs vLLM + GPU CI) |
| Table III PPL / MMLU | — | Planned |
| Multi-model Table II | Partial: gpt2 measured + Qwen cited external | **Partial** (`paper_results.json`, `external_baselines.json`) |
| Measured harness Table | `paper_results.json` (gpt2); Qwen from `external_baselines.json` | **gpt2 CPU** + **cited external** |
| Prefix TTFT Fig (2048) | `sharegpt_2048_sim.json` | **KV sim measured** |

## Quick commands

```bash
# Python-only MVP-A (no MLIR build required)
pytest tests/test_mvp_a_e2e.py -m "not network" -q

# MVP-C torch GPU KV path
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
| `mvp_evaluation_nature` | **Yes** (panels a–c) | MVP-A/B/C CI + docs |
| `prefix_cache_nature` | **Partial** | `sharegpt_prefix_bench` |
| `block_size_optimization_nature` | **Partial** | Algorithm 1 + illustrative sweep |
| `multi_model_comparison_nature` | **No** (projected) | Table II targets |
| `attention_speedup_nature` | **No** (micro-bench) | Illustrative arrays |

## Honesty notes

- Throughput heatmap / Table II remain **projected** until GPU harness feeds JSON.
- MVP-A proves **compile-time block sizing + single-layer IR lowering + numeric reference**; it does not claim vLLM-scale serving wins.
