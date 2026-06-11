# LLMIR Paper Revision Notes (ICCD 2025 feedback)

Maps reviewer concerns to **what is verified in the repository** vs **what remains planned or illustrative**.

Authoritative status: [`docs/PAPER_REVISION_TRACEABILITY.md`](../docs/PAPER_REVISION_TRACEABILITY.md), [`docs/CAPABILITY_MATRIX.md`](../docs/CAPABILITY_MATRIX.md), and long-term bar [`docs/PAPER_TOP_TIER_BAR.md`](../docs/PAPER_TOP_TIER_BAR.md) (OSDI/ASPLOS/MLSys target vs current ICCD honest scope).

## Verified in open source (cite these)

| Reviewer theme | Response in paper | Repo evidence |
|----------------|-------------------|---------------|
| Implementation / compile path | §4, Algorithm 1, **E1** | `BlockSizeAnalysis.cpp`, `tests/test_mvp_a_e2e.py`, `gpt2_e1_snippet.mlir` |
| Model → IR → runtime | §3.1 pipeline, E1 lowering | `llmir-compile --mvp-a-e2e`, lit tests |
| Prefix / shared-prefix decoder workload | **E2**, Fig. prefix TTFT | `sharegpt_prefix_bench.py`, `shared_prefix_decoder_2048_sim.json`, `paper_results.json` |
| GPU KV without NumPy round-trip | **E3** | `TorchGpuPagedKVCache`, `tests/test_mvp_c_e2e.py` |
| Measured decode baseline | gpt2 HF vs `llmir-paged` | `paper_results.json` |
| External serving reference | Qwen vLLM cited row | `external_baselines.json` (Qwen official benchmark) |
| Compositional trace (E1+E2+E3) | §5 E4 | `e4_compositional.json`, `scripts/e4_compositional_verify.py` |
| Verifiable layer ablations | §5 E5 | `e5_ablation.json`, `scripts/e5_ablation_verify.py` |
| Multi-backend decode parity | §5 E6 | `e6_backend_parity.json`, `scripts/e6_backend_parity_verify.py` |
| Lowered hot path semantics | §5 M5 | `m5_lowered_hot_path.json`, `scripts/m5_lowered_hot_path_verify.py` |
| CPU artifact bundle | §5 M6 | `artifact_manifest.json`, `scripts/verify_artifact_bundle.py`, `reproduce_paper.sh` |

## Partially addressed (labeled illustrative / projected in `revised.tex`)

| Reviewer theme | Paper location | Status |
|----------------|----------------|--------|
| Multi-model throughput (8 models × 5 frameworks) | Appendix A (design targets), main text Table measured_harness only | **No LLMIR GPU harness JSON**; gpt2 measured + Qwen cited only |
| Quality (PPL / MMLU) | Table III | **Planned** — illustrative targets, not measured |
| Memory config / block sweep | Table IV, block-size figure | **KV-append microbench lineage** — not A100 LLaMA e2e |
| Multi-GPU scaling | Table V | **Illustrative** — no artifact |
| Ablation | Table VI | **Illustrative throughput** — verifiable E5 proxy in `e5_ablation.json` (not Table VI numbers) |
| Attention speedup figure | Fig. attention | **Standalone C++ microbench** — not production Flash on hot path |
| TensorRT / MLC / SGLang columns | Table II | **Design targets** until `llmir-benchmark` + GPU CI |

## Reviewer concern checklist

### Review 1 — implementation & experimental detail

- **Implementation depth**: Addressed in text + E1 (**verified**).
- **Experimental setup (A100, decoder length sweeps, MMLU)**: §5.1 splits **target environment** vs **completed measurements**; A-class CI covers E1–E6 + M5/M6 on CPU; optional E8 GPU bench is B-class only (`e8_empirical_gpu.json`).

### Review 2 — IR flow, multi-model, kernel selection

- **IR flow**: Addressed (**verified** E1).
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
- Committed raw benchmark logs / plots under `benchmark/**/results/` — **removed**; regenerate via harness scripts. `benchmark_summary.txt` retained with scope banner for appendix lineage.

## Figure generation

```bash
# Verified / measured figures only (default for paper refresh)
python3 IEEE-conference/figures/generate_all_nature_figures.py

# Illustrative / projected figures (Table II heatmap, block sweep, attention microbench)
python3 IEEE-conference/figures/generate_projected_figures.py

# E4/E5 multi-bucket appendix tables (from committed JSON)
python3 scripts/generate_paper_bucket_tables_tex.py
```

## Compile revised paper

```bash
cd IEEE-conference
pdflatex LLMIR-paper-ICCD2025-revised.tex
pdflatex LLMIR-paper-ICCD2025-revised.tex
```
