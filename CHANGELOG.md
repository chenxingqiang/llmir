# Changelog

All notable changes to LLMIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Lab gates**: `verify_lab_gates.py` integrated into walkthrough/prepare_release checks.
- **PyPI republish workflow**: `pypi-republish.yml` for maintainer retry after trusted publisher setup.
- **Lab evidence wiring**: `lab_status_summary` module in walkthrough + evidence dashboard; CI uploads lab JSON.
- **Lab smoke hub**: `lab_smoke_all.sh`, `lab_status_summary.py`, `check_native_build_prereqs.sh`, `docs/MLIR_NATIVE_BUILD.md`.
- **E8 GPU lab smoke**: `e8_lab_smoke.sh` + `e8-gpu-lab.yml` (`require_completed` for strict CUDA lab).
- **Walkthrough closure**: PyPI verify step in `walkthrough_a_class.sh`; walkthrough gates log pending PyPI.
- **MLIR lit lab CI**: `mlir-lit-lab.yml` workflow_dispatch; `mlir_lit_smoke.sh` exits 0 when skipped.

### Added (R22)

- **PyPI release verify**: `verify_pypi_release.py`, `pypi_release_status.json`, dashboard row; `docs/PYPI_TRUSTED_PUBLISHER.md` for OIDC publisher setup.

## [0.2.2] - 2026-06-12

### Fixed

- **CI Python 3.9/3.10**: `scripts/pyproject_tools.py` with `tomli` fallback (no stdlib `tomllib`).
- **Release tag dry-run**: succeeds when the release tag already exists (gates-only validation).
- **mypy lint**: type fixes in `native_bridge`, `prefix_kv_store`, `engine` (vLLM lazy init).

## [0.2.1] - 2026-06-05

### Added

- **Release tag automation**: `scripts/tag_release.sh`, workflow `release-tag.yml` (push `v*` → PyPI via `python-package.yml`).

### Fixed

- MVP-C `test_mvp_c_engine_device`: CPU CI no longer calls `model.to(cuda)` when CUDA is mocked; GPU integration test gated with `skipif`.

### Added

- **Release prep**: `scripts/prepare_release.sh`, `docs/PYPI_RELEASE_CHECKLIST.md`, workflow `release-prep.yml`.
- **MLIR lit lab**: `scripts/build_mlir_opt.sh`, `scripts/mlir_lit_smoke.sh`, `docs/MLIR_LIT_RUNBOOK.md`.
- **Evidence chain (Loops R6–R18)**: E4–E6 multi-bucket traces, M6 artifact bundle extensions, E8 B-class scaffold, A-class walkthrough CI, evidence dashboard, lint gates, MLIR lit catalog, paper appendix bucket tables.
- **CUDA kernels (P5)**: `CUDAKernels.cu`, `cuda_probe`, native ctypes bindings, `build_native_runtime.sh`, `package_native_lib.sh`.
- **vLLM KV connector (P4)**, GPU compare benchmarks, prefix cache on `llmir_paged`.
- **Benchmark harness (P3)**: `llmir-benchmark --compare`, prefix microbenches, `native-runtime.yml`.

### Changed

- **Python**: minimum supported version is **3.9** (CI matrix 3.9–3.12; 3.8 dropped). See `docs/PYTHON_VERSION_POLICY.md`.

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
