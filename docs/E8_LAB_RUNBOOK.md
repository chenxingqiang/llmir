# E8 Lab Runbook (B-Class GPU)

Run **E8** on a machine with CUDA to produce `status=completed` in `e8_empirical_gpu.json`. CPU CI correctly writes `skipped`; do not treat that as failure.

## Prerequisites

- NVIDIA GPU + driver
- `pip install -e ".[full]"` (transformers, torch with CUDA)
- Optional: vLLM if comparing `vllm` backend

## One-shot lab script

```bash
export E8_MODEL=gpt2
export E8_BACKENDS=hf,llmir-paged
bash scripts/e8_lab_run.sh
```

On success: `e8_empirical_gpu.json` has `status=completed` and non-empty `results`.

## Manual steps

```bash
pytest tests/test_e8_empirical_gpu.py -q
python3 scripts/e8_empirical_gpu_bench.py --model gpt2 --backends hf,llmir-paged
python3 scripts/verify_artifact_bundle.py --skip-figures
```

## GitHub Actions (optional)

- **CPU walkthrough:** `.github/workflows/a-class-walkthrough.yml` (auto on PR/main)
- **E8 dispatch:** Actions → *E8 empirical GPU* → `workflow_dispatch`

Upload artifact `e8-empirical-gpu.json` after GPU runner completes.

## Paper wording

Only cite throughput rows when `status=completed` and hardware SKU is documented in the run log. Never mix E8 with E4 compositional claims.
