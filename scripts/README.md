# Scripts

Reproducible benchmarks and paper evaluation harnesses (E1–E3) for LLMIR. Run from repository root.

## Paper / CI harnesses

| Script | Description |
|--------|-------------|
| `paper_benchmark_collect.py` | Writes `IEEE-conference/benchmarks/paper_results.json` |
| `sharegpt_prefix_bench.py` | E2 shared-prefix decoder workload (legacy filename) |
| `plot_from_results.py` | Plots benchmark JSON (no hard-coded throughput) |
| `cpu_inference_compare.py` | CPU HF vs `llmir-paged` vs optional vLLM |
| `gpu_inference_compare.py` | GPU-oriented compare wrapper (CI workflow) |

## E1 / E3 / native runtime

| Script | Description |
|--------|-------------|
| `mvp_a_single_layer_e2e.sh` | E1 single-layer compile + reference |
| `mvp_c_cuda_kv_bench.py` | E3 CUDA KV microbench |
| `build_native_runtime.sh` | Build `libMLIRLLMRuntime` |
| `package_native_lib.sh` | Package native runtime for pip path |
| `kv_backend_compare.py` | NumPy vs native KV microbench |

## Prefix / integration smoke

| Script | Description |
|--------|-------------|
| `prefix_cache_benchmark.py` | Prefix cache JSON benchmark |
| `prefix_prefill_e2e.py` | Prefix prefill end-to-end JSON |
| `vllm_kv_connector_smoke.py` | vLLM KV connector smoke test |

## Optional local validation

| Script | Description |
|--------|-------------|
| `mps_full_pipeline.py` | Apple Silicon validation pipeline |

GPU Llama-3.1 experimental baseline: see [`benchmark/LLM/README_LLAMA31.md`](../benchmark/LLM/README_LLAMA31.md).
