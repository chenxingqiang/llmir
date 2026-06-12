# E8 Lab Runbook (B-Class GPU)

Run **E8** on a machine with CUDA to produce `status=completed` in `e8_empirical_gpu.json`. CPU CI correctly writes `skipped`; do not treat that as failure.

## Prerequisites

- NVIDIA GPU + driver
- `pip install -e ".[full]"` (transformers, torch with CUDA)
- Optional: vLLM if comparing `vllm` backend

## Smoke (CPU or GPU — honest skip)

```bash
bash scripts/e8_lab_preflight.sh
bash scripts/e8_lab_smoke.sh
python3 scripts/verify_e8_lab.py
```

Writes `e8_empirical_gpu.json` with `status=skipped` (no CUDA) or `completed` (GPU).
Exit 0 for both; used by A-class walkthrough and CPU CI.

## Strict lab (GPU required)

```bash
bash scripts/e8_lab_preflight.sh --strict
export E8_MODEL=gpt2
export E8_BACKENDS=hf,llmir-paged
bash scripts/e8_lab_run.sh
python3 scripts/verify_e8_lab.py --require-completed
```

On success: `status=completed` and non-empty `results`. Fails if CUDA absent.

## Manual steps

```bash
pytest tests/test_e8_empirical_gpu.py -q
python3 scripts/e8_empirical_gpu_bench.py --model gpt2 --backends hf,llmir-paged
python3 scripts/verify_artifact_bundle.py --skip-figures
```

## GitHub Actions (optional)

- **CPU walkthrough:** `.github/workflows/a-class-walkthrough.yml` (auto on PR/main)
- **E8 GPU lab:** Actions → *E8 GPU lab (optional)* → `workflow_dispatch`
  - Default: smoke mode (`require_completed=false`) — honest skip on `ubuntu-latest`
  - GPU self-hosted: set `require_completed=true` on a CUDA runner (preflight `--strict`)
- Legacy: `e8-empirical-gpu.yml` (same smoke path)

Upload artifact `e8-empirical-gpu.json` after the workflow completes.

## Paper wording

Only cite throughput rows when `status=completed` and hardware SKU is documented in the run log. Never mix E8 with E4 compositional claims.
