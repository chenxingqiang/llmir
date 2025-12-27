# Changelog

All notable changes to LLMIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
