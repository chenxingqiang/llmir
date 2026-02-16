# Scripts

Benchmark and utility scripts for LLMIR.

| Script | Description |
|--------|-------------|
| `comprehensive_benchmark.sh` | Compare PyTorch, vLLM, and SGLang inference performance |
| `quick_benchmark.sh` | Quick benchmark for Qwen models |
| `run_real_benchmark.sh` | Run real model benchmarks |
| `vllm_comparison.sh` | vLLM comparison benchmarks |
| `llmir_model_benchmark.py` | Python benchmark for LLMIR KV cache and model optimization |
| `real_model_benchmark.py` | Benchmark with real LLM models |
| `docker_run_benchmark.sh` | Run Llama 3.1 benchmark in Docker (GPU required) |
| `mps_full_pipeline.py` | MPS validation: config, KV micro-bench, model.generate, **LLMIR E2E** (KV round-trip) |

**MPS (Apple Silicon)**: `python scripts/mps_full_pipeline.py` validates: (1) LLMIR config, (2) model load, (3) PagedKVCache micro-bench with synthetic data, (4) transformers inference, (5) **LLMIR E2E**—manual generation with KV routed through external store (round-trip through numpy); compares against model.generate() for correctness.

**Note**: Run scripts from the project root. For Docker benchmarks, ensure `HUGGINGFACE_TOKEN` is set for gated models.
