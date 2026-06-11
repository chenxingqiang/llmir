# Changelog

All notable changes to LLMIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- MVP-C `test_mvp_c_engine_device`: CPU CI no longer calls `model.to(cuda)` when CUDA is mocked; GPU integration test gated with `skipif`.

### Added

- **Release prep**: `scripts/prepare_release.sh`, `docs/PYPI_RELEASE_CHECKLIST.md`, workflow `release-prep.yml`.

### Changed

- **Python**: minimum supported version is **3.9** (CI matrix 3.9–3.12; 3.8 dropped). See `docs/PYTHON_VERSION_POLICY.md`.

### Added (Loop R6–R16 evidence chain)

- E4–E6 multi-bucket traces, M6 artifact bundle extensions, E8 B-class scaffold.
- A-class walkthrough + CI (`a-class-walkthrough.yml`), evidence dashboard, lint gates.
- MLIR lit catalog (`decoder_workload_buckets.mlir`), paper appendix tables from JSON.

### Added (P5)

- **CUDA kernels**: `CUDAKernels.cu` wired into `MLIRLLMRuntime` via `LLMIR_ENABLE_CUDA`.
- MQA / GQA / pruned attention launchers implemented (MVP block kernel).
- C API: `llmir_cuda_runtime_available`, `llmir_cuda_device_count`.
- Python: `llmir.runtime.cuda_probe`, native ctypes bindings.
- `scripts/build_native_runtime.sh` auto-enables CUDA when `nvcc` is present.
- `scripts/package_native_lib.sh` for dev wheel bundling; `docs/P5_CUDA_NATIVE.md`.

### Added (P4)

- **vLLM KV connector**: `LLMIRConnector` + `LLMIRKVStorage` (`llmir.integration.vllm_connector`).
- `register_llmir_vllm_connector()` for vLLM V1 factory registration.
- **GPU compare**: `--compare-device` / `--compare-dtype` on `llmir-benchmark --compare`.
- `scripts/gpu_inference_compare.py`, `scripts/vllm_kv_connector_smoke.py`.
- Optional workflow `.github/workflows/gpu-benchmark.yml` (workflow_dispatch).

### Added (prefix on `llmir_paged`)

- `PrefixKVStore` + `PagedKVDecoder.warm_prefix` / decode-time longest-prefix restore.
- `DecodeResult.prefix_hit_tokens` / `prefill_tokens_computed`; surfaced on `LLMEngine` metrics.
- `scripts/prefix_prefill_e2e.py`; tests `test_prefix_kv_store.py`, `test_paged_decoder_prefix.py`.

### Added (P3)

- `llmir.benchmark`: `run_inference_compare`, prefix cache microbenches.
- `llmir-benchmark --compare hf,vllm,llmir-paged` for reproducible E2E JSON.
- `llmir-benchmark --prefix-bench` and `scripts/prefix_cache_benchmark.py`.
- `scripts/build_native_runtime.sh` + optional CI workflow `native-runtime.yml`.
- `scripts/cpu_inference_compare.py` now delegates to `llmir.benchmark`.

## [0.2.0] - 2026-06-05

### Added (P2 MVP)

- **`llmir-compile`** CLI: emit KV micro-pipeline MLIR, optional `mlir-opt`, reference run.
- **`llmir.compiler`**: `emit_kv_micro_pipeline_mlir`, `run_mlir_opt`, reference interpreter.
- **`llmir.importers.toy_attention`**: trace toy SDPA → MLIR.
- **`examples/e2e/kv_micro_pipeline.py`**: one-command numerical check vs PyTorch SDPA.
- **`scripts/kv_backend_compare.py`**: NumPy vs native KV microbench.
- **`tests/test_compile_e2e.py`**, **`tests/test_e2e_token_consistency.py`** (network).

### Changed (P0/P1 from 0.2.0 prep)

- **Breaking (serving):** Default `LLMEngine` / `EngineConfig` backend is now
  `llmir_paged` (real HuggingFace inference). `backend="llmir"` remains for
  scheduler smoke tests and emits `UserWarning`.
- `PagedKVDecoder` uses `create_paged_kv_cache()` — prefers C++
  `libMLIRLLMRuntime` when `LLMIR_LIB_PATH` is set, else NumPy reference.
- `llmir-benchmark` CLI description clarifies KV **microbenchmark** vs e2e inference.

### Added

- `docs/CAPABILITY_MATRIX.md` — honest feature status table.
- `llmir[native]` optional extra (documents C++ runtime dependency).
- `llmir.runtime.native_bridge`, `native_kvcache`, `kv_factory`.
- `scripts/plot_from_results.py` — plots JSON from benchmarks (no hard-coded throughput).
- `examples/demos/simulated/` and `IEEE-conference/figures/paper-only/` READMEs.

### Moved

- `examples/demo_llmir_0.6b.py` → `examples/demos/simulated/`
- Hard-coded paper chart scripts → `IEEE-conference/figures/paper-only/`

## [0.1.0] - 2025-12-26

### Added

#### Core Features
- **PagedKVCache**: Block-based KV cache with dynamic memory management
- **QuantizedKVCache**: INT8/INT4 quantization support for 4-8x memory reduction
- **DistributedKVCache**: Multi-GPU sharding with layer/head/sequence-wise strategies
- **SpeculativeKVCache**: KV cache branching for speculative decoding
- **PrefixCache**: Radix tree-based prefix caching for prompt reuse

#### Serving
- **ContinuousBatchingEngine**: vLLM-style dynamic batch management
- **LLMEngine**: High-level engine with vLLM-compatible API
- **SamplingParams**: Comprehensive generation parameters
- **SchedulerConfig**: Configurable scheduling policies (FCFS, Priority, Adaptive)

#### Profiling
- **Profiler**: Main profiler with tracing and event recording
- **MemoryProfiler**: Memory usage tracking
- **LatencyProfiler**: Latency statistics with percentiles (P50/P90/P95/P99)
- **ThroughputMonitor**: Tokens/second measurement
- Chrome trace export support

#### Model Optimizations
- **LlamaOptimizer**: Support for Llama 1/2/3/3.1 models (7B to 405B)
- **MistralOptimizer**: Support for Mistral 7B and Mixtral 8x7B/8x22B
- **PhiOptimizer**: Support for Phi-2 and Phi-3 variants
- **ModelRegistry**: Pre-configured model presets
- **ModelMemoryEstimator**: Memory usage estimation and planning

#### CLI Tools
- `llmir-profile`: Performance profiling command
- `llmir-benchmark`: Benchmarking command

### Configuration Classes
- `KVCacheConfig`: KV cache configuration
- `QuantizationConfig`: Quantization settings
- `SpeculativeConfig`: Speculative decoding settings
- `ShardingConfig`: Multi-GPU sharding settings
- `SchedulerConfig`: Scheduler configuration
- `EngineConfig`: LLM engine configuration

## Links
- [GitHub Repository](https://github.com/chenxingqiang/llmir)
- [Documentation](https://chenxingqiang.github.io/llmir-www/)
