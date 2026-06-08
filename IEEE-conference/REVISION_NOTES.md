# LLMIR Paper Revision Notes (ICCD 2025 feedback)

Maps reviewer concerns to **what is verified in the repository** vs **what remains planned or illustrative**.

Authoritative status: [`docs/PAPER_REVISION_TRACEABILITY.md`](../docs/PAPER_REVISION_TRACEABILITY.md) and [`docs/CAPABILITY_MATRIX.md`](../docs/CAPABILITY_MATRIX.md).

## Verified in open source (cite these)

| Reviewer theme | Response in paper | Repo evidence |
|----------------|-------------------|---------------|
| Implementation / compile path | §4, Algorithm 1, MVP-A | `BlockSizeAnalysis.cpp`, `tests/test_mvp_a_e2e.py`, `gpt2_mvp_a_snippet.mlir` |
| Model → IR → runtime | §3.1 pipeline, MVP-A lowering | `llmir-compile --mvp-a-e2e`, lit tests |
| Prefix / ShareGPT workload | MVP-B, Fig. prefix TTFT | `sharegpt_prefix_bench.py`, `sharegpt_2048_sim.json`, `paper_results.json` |
| GPU KV without NumPy round-trip | MVP-C | `TorchGpuPagedKVCache`, `tests/test_mvp_c_e2e.py` |
| Measured decode baseline | gpt2 HF vs `llmir-paged` | `paper_results.json` |
| External serving reference | Qwen vLLM cited row | `external_baselines.json` (Qwen official benchmark) |

## Partially addressed (labeled illustrative / projected in `revised.tex`)

| Reviewer theme | Paper location | Status |
|----------------|----------------|--------|
| Multi-model throughput (8 models × 5 frameworks) | Appendix A (design targets), main text Table measured_harness only | **No LLMIR GPU harness JSON**; gpt2 measured + Qwen cited only |
| Quality (PPL / MMLU) | Table III | **Planned** — illustrative targets, not measured |
| Memory config / block sweep | Table IV, block-size figure | **KV-append microbench lineage** — not A100 LLaMA e2e |
| Multi-GPU scaling | Table V | **Illustrative** — no artifact |
| Ablation | Table VI | **Illustrative** — no artifact |
| Attention speedup figure | Fig. attention | **Standalone C++ microbench** — not production Flash on hot path |
| TensorRT / MLC / SGLang columns | Table II | **Design targets** until `llmir-benchmark` + GPU CI |

## Reviewer concern checklist

### Review 1 — implementation & experimental detail

- **Implementation depth**: Addressed in text + MVP-A (**verified**).
- **Experimental setup (A100, ShareGPT, C4, MMLU)**: §5.1 now splits **target environment** vs **completed measurements**; only MVP + gpt2 CPU + KV sim are in CI today.

### Review 2 — IR flow, multi-model, kernel selection

- **IR flow**: Addressed (**verified** MVP-A).
- **Multi-model experiments**: **Projected** in Table II until GPU harness lands.
- **Pool+Unified / hybrid GPU text**: Retained as **design rationale** paired with illustrative tables.

### Review 3 — compile-time vs runtime

- **Table I compile vs runtime**: Addressed in prose.
- **Figure readability**: Nature-style regeneration; measured vs projected figures use separate generator entry points.

### Review 4 — breadth & quality & TRT

- **More model families**: Table II rows are **targets**, not measured LLMIR runs.
- **PPL / MMLU**: Table III **planned** — do not cite as results until harness exists.
- **TensorRT-LLM column**: Illustrative comparison only.

## Deprecated artifacts (do not use)

- `LLMIR-paper-ICCD2025-anonymous.tex` — **removed** (unverified throughput claims).
- Legacy matplotlib v1/v2 figure scripts — **removed** (use `create_*_nature.py` + generators above).
- HF-only GPU scripts (`run_real_benchmark.sh`, `comprehensive_benchmark.sh`, etc.) — **removed**; use `cpu_inference_compare.py` / `gpu_inference_compare.py` / `paper_benchmark_collect.py`.
- `IEEE-conference/figures/paper-only/` — hard-coded arrays; run only via `generate_projected_figures.py`.

## Figure generation

```bash
# Verified / measured figures only (default for paper refresh)
python3 IEEE-conference/figures/generate_all_nature_figures.py

# Illustrative / projected figures (Table II heatmap, block sweep, attention microbench)
python3 IEEE-conference/figures/generate_projected_figures.py
```

## Compile revised paper

```bash
cd IEEE-conference
pdflatex LLMIR-paper-ICCD2025-revised.tex
pdflatex LLMIR-paper-ICCD2025-revised.tex
```
