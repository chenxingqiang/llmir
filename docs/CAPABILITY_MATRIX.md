# LLMIR capability matrix

Legend:

| Status | Meaning |
|--------|---------|
| **C++** | Implemented in `lib/Dialect/LLM/`; validated by C++ tests / lit |
| **Python (ref)** | Reference implementation in `src/llmir/` (often NumPy) |
| **Python (native)** | Python API backed by `libMLIRLLMRuntime` when built + `LLMIR_LIB_PATH` |
| **Planned** | Design or partial code, not on default user path |
| **Demo only** | Simulation / paper figures, not production |

## Serving (`LLMEngine`)

| Feature | Status | Notes |
|---------|--------|-------|
| `backend="llmir_paged"` | **Python (ref)** / **Python (native)** | Default; HF forward + per-layer KV via `create_paged_kv_cache` |
| `backend="vllm"` | Pass-through | vLLM owns hot path |
| `backend="llmir"` | **Demo only** | Placeholder tokens; warns on construction |
| Continuous batching + real model | **Planned** | Scheduler exists; not wired to HF/vLLM yet |

## KV cache

| Feature | Status | Notes |
|---------|--------|-------|
| Block-paged allocator | **C++** | `Runtime/KVCache.cpp` |
| `PagedKVCache` (pip default) | **Python (ref)** | Dict/list store; use `LLMIR_KV_BACKEND=numpy` |
| `PagedKVCache` (optional) | **Python (native)** | `pip install llmir[native]` + built `.so` |
| INT8/INT4 KV | **C++** / **Python (ref)** | Real quant in C++; Python returns ratio constants only |
| Prefix cache (radix) | **Python (ref)** | `PrefixCache` + `llmir-benchmark --prefix-bench`; C++ in `PrefixCache.cpp` |
| Prefix cache on `llmir_paged` | **Python (ref)** | `PagedKVDecoder.warm_prefix` + `PrefixKVStore`; metrics on `RequestOutput` |
| `llmir-benchmark --compare` | **Python (ref)** | E2E `hf,vllm,llmir-paged` JSON compare |
| Distributed KV | **Planned** | Python shard-0 only |
| Speculative KV branches | **Planned** | Python `verify()` stub |

## Compiler / MLIR

| Feature | Status | Notes |
|---------|--------|-------|
| `llm` dialect ops & types | **C++** | Lit tests under `test/Dialect/LLM/` |
| Optimization passes | **C++** | Partial pipeline; some passes commented out |
| PyTorch → MLIR import | **Python (ref)** | `llmir.importers` + toy SDPA; emits MLIR text |
| `llmir-compile` KV micro-pipeline | **Python (ref)** | Emit + optional `mlir-opt` + NumPy/native reference |
| `mlir-opt` → execute e2e | **Planned** | Lowering works; full execution JIT not wired |
| CUDA Flash kernels | **Planned** | `CUDAKernels.cu` not in CMake build |

## Benchmarks & docs

| Artifact | Status | Notes |
|----------|--------|-------|
| `llmir-benchmark` CLI | **Python (ref)** | KV append microbench + model **config** |
| `scripts/cpu_inference_compare.py` | **Python (ref)** | E2E CPU compare (HF / vLLM / llmir_paged) |
| `IEEE-conference/figures/paper-only/` | **Demo only** | Hard-coded chart data |
| `IEEE-conference/A800_GPU_TEST_RESULTS.md` | Environment-specific | Re-run on target GPU to reproduce |

## Versioning

- **0.1.x**: Experimental; matrix above applies.
- **0.2.0 target (MVP)**: native KV on pip path, `llmir_paged` CI token gate, one MLIR e2e example, bench JSON baselines.
