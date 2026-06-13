# Paper revision traceability (ICCD 2025 → code)

Maps reviewer-facing claims in `IEEE-conference/REVISION_NOTES.md` to **verifiable** artifacts in this repository.

## Paper experiment naming (E1–E6)

| Paper ID | Full name | Legacy CLI / tests | Doc |
|----------|-----------|-------------------|-----|
| **E1** | Compile-Time Pass Verification | `llmir-compile --mvp-a-e2e`, `tests/test_mvp_a_e2e.py` | `docs/E1_COMPILE_PASS_VERIFICATION.md` |
| **E2** | Prefix-Aware Serving Evaluation | `llmir-benchmark --shared-prefix-bench`, `scripts/sharegpt_prefix_bench.py` | `docs/E2_PREFIX_SERVING_EVAL.md`, `docs/DECODER_WORKLOAD_ARCHITECTURES.md` |
| **E3** | GPU-Resident KV Integration | `llmir-benchmark --mvp-c-bench`, `tests/test_mvp_c_e2e.py` | `docs/E3_GPU_KV_INTEGRATION.md` |
| **E4** | Compositional / Trace-Driven Verification | `scripts/e4_compositional_verify.py`, `tests/test_e4_compositional.py` | `docs/E4_COMPOSITIONAL_VERIFICATION.md` |
| **E5** | Ablation at Verifiable Layers | `scripts/e5_ablation_verify.py`, `tests/test_e5_ablation.py` | `docs/E5_ABLATION.md` |
| **E6** | Multi-Backend Correctness Parity | `scripts/e6_backend_parity_verify.py`, `tests/test_e6_backend_parity.py` | `docs/E6_BACKEND_PARITY.md` |
| **E8** | Empirical GPU Benchmark (optional) | `scripts/e8_empirical_gpu_bench.py`, `tests/test_e8_empirical_gpu.py` | `docs/E8_EMPIRICAL_GPU_BENCH.md` |

Repository code and CI retain legacy `mvp-*` flag names; the paper uses **E1–E6** for A-class claims and **E8** only for optional B-class GPU rows.

| Revision item | Verification | Status |
|---------------|--------------|--------|
| §4 Algorithm 1 block size optimization | `lib/Dialect/LLM/Transforms/BlockSizeAnalysis.cpp`, `src/llmir/compiler/block_size.py` | Implemented |
| `llm-optimize-kv-cache` applies block size | `KVCacheOptimization.cpp` calls `applyBlockSizeOptimizationToFunc` | Implemented |
| Lit: block size rewrite | `test/Dialect/LLM/kv_cache_optimization.mlir`, `mvp_single_layer_pipeline.mlir` | Implemented |
| §3.1 model → IR → kernel (single layer) | `llmir-compile --mvp-a-e2e`, `src/llmir/compiler/mvp_pipeline.py` | **E1** |
| Lower to runtime calls | `-llm-lower-kv-cache-ops` → `@mlir_llm_*` | Implemented (needs `llmir-opt`) |
| M5 lowered hot path execution | `m5_lowered_hot_path.json`, `scripts/m5_lowered_hot_path_verify.py` | **Semantic parity vs reference** |
| M6 artifact bundle | `artifact_manifest.json`, `artifact_bundle_status.json`, `scripts/verify_artifact_bundle.py` | **Manifest + CPU regen** |
| Reference correctness | `tests/test_mvp_a_e2e.py`, `tests/test_compile_e2e.py` | CI (Python) |
| §5 shared-prefix decoder / TTFT workload | `scripts/sharegpt_prefix_bench.py`, `llmir-benchmark --shared-prefix-bench` | **E2** (sim + llmir_paged E2E) |
| GPU KV without CPU NumPy round-trip | `TorchGpuPagedKVCache`, `PagedKVDecoder` GPU path, `llmir-benchmark --mvp-c-bench` | **E3** |
| Native CUDA KV kernels | `libMLIRLLMRuntime` + `cuda_probe`, `LLMIR_KV_BACKEND=native` | **E3** (optional build) |
| §5 evaluation scope | Main text: compiler verification + serving proxies only | **No operator-level claims** |
| Operator / FlashAttention speedup | Appendix `app:future_ops` only | **Not LLMIR kernels** — `benchmark/attention/` toys |
| Main-text throughput | Table measured_harness only | **gpt2 CPU** + **cited Qwen** |
| Appendix scale-out | `app:projected` design targets | **Illustrative** |
| Prefix TTFT Fig (2048) | `shared_prefix_decoder_2048_sim.json` | **KV sim measured** |
| E4 compositional (E1+E2+E3 trace) | `e4_compositional.json`, `scripts/reproduce_paper.sh` | **Analytical + E2 sim bound** |
| E4 multi-bucket (S1/S2/S3) | `e4_compositional_buckets.json`, `scripts/e4_e5_multi_bucket_verify.py` | **Per-bucket ideal-bound check** |
| E5 ablation switches | `e5_ablation.json` | **Isolated + cumulative proxy deltas** |
| E5 multi-bucket (S1/S2/S3) | `e5_ablation_buckets.json`, `scripts/e4_e5_multi_bucket_verify.py` | **Per-bucket switch matrix** |
| Appendix E4/E5 bucket tables | `IEEE-conference/generated/e4_e5_bucket_tables.tex`, `scripts/generate_paper_bucket_tables_tex.py` | **LaTeX from JSON** |
| E6 backend parity | `e6_backend_parity.json` | **Decode tokens + KV micro match** |

## Quick commands

```bash
# Python-only E1 (no MLIR build required)
pytest tests/test_mvp_a_e2e.py -m "not network" -q

# E3 torch GPU KV path
pytest tests/test_torch_gpu_kv_cache.py tests/test_mvp_c_e2e.py -m "not network" -q
llmir-benchmark --mvp-c-bench --model gpt2 -o mvp_c.json

# E4 compositional (CPU, no GPU)
pytest tests/test_e4_compositional.py -q
python3 scripts/e4_compositional_verify.py --from-sim \
  IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json

# Paper measured JSON + figures
python3 scripts/paper_benchmark_collect.py --model gpt2
python3 IEEE-conference/figures/create_measured_figures_nature.py

# E5 ablation (CPU)
pytest tests/test_e5_ablation.py -q
python3 scripts/e5_ablation_verify.py --from-sim \
  IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json

# E4/E5 multi-bucket (S1/S2/S3)
pytest tests/test_e4_e5_multi_bucket.py -q
python3 scripts/e4_e5_multi_bucket_verify.py

# E6 backend parity (offline toy model)
pytest tests/test_e6_backend_parity.py -q
python3 scripts/e6_backend_parity_verify.py --model toy

# M6 artifact bundle verify
python3 scripts/verify_artifact_bundle.py
pytest tests/test_artifact_bundle.py -q

# E8 optional GPU bench (skips without CUDA)
bash scripts/e8_lab_smoke.sh

# Lab rollup (mlir + E8 + PyPI + native prereqs)
bash scripts/lab_smoke_all.sh
python3 scripts/verify_lab_gates.py
cat IEEE-conference/benchmarks/lab_status_summary.json

# Fast reviewer walkthrough (verify committed artifacts, ~5 min CPU)
bash scripts/walkthrough_a_class.sh

# CI: .github/workflows/a-class-walkthrough.yml (same script on ubuntu-latest)
# Lab CI: .github/workflows/lab-smoke.yml (workflow_dispatch)

# E8 GPU strict lab (B-class, requires CUDA)
bash scripts/e8_lab_run.sh   # status=completed on GPU lab

# PyPI / release (maintainer)
bash scripts/prepare_release.sh
bash scripts/tag_release.sh --dry-run
# See docs/PYPI_TRUSTED_PUBLISHER.md, docs/CI_WORKFLOW_INDEX.md

# Full A-class reproduce (E1–E6 + M5 + figures + lab tail)
bash scripts/reproduce_paper.sh

# Full path when llmir-opt is on PATH
llmir-compile --mvp-a-e2e --run-opt --run-reference --compare-torch \
  --seq-len 8 --mvp-json /tmp/mvp_a.json -o /tmp/mvp_a.mlir

# MLIR lit suite (optional mlir-opt)
bash scripts/mlir_lit_smoke.sh
# Build: bash scripts/build_mlir_opt.sh (see docs/MLIR_NATIVE_BUILD.md)
```

## Paper figures (Nature style)

Regenerate measured: `python3 IEEE-conference/figures/generate_all_nature_figures.py`

Regenerate projected/illustrative: `python3 IEEE-conference/figures/generate_projected_figures.py`

| Figure | Verified? | Source |
|--------|-----------|--------|
| `e1_e3_evaluation_nature` | **Yes** (a,b from JSON; c illustrative) | `e4_compositional_buckets.json`, `shared_prefix_decoder_2048_sim.json`, E6 parity |
| `prefix_cache_nature` | **Partial** | `sharegpt_prefix_bench` |
| `block_size_optimization_nature` | **Partial** | Algorithm 1 + illustrative sweep |
| `multi_model_comparison_nature` | **No** (projected) | Appendix A design targets |
| `attention_speedup_nature` | **No** (future work) | Appendix `app:future_ops`; not LLMIR lowered kernels |

## Honesty notes

- Main text §5: **E1–E6 analytical harnesses + M6 CPU bundle**; serving proxies (E2/E3) and no operator-level kernel claims.
- **E8** (optional): B-class GPU throughput only; `status=skipped` on CPU CI is expected and honest.
- Throughput heatmap / scale-out tables: **projected** in Appendix `app:projected`.
- Attention figure: **future operator work** in Appendix `app:future_ops`; not evidence for LLMIR codegen today.
