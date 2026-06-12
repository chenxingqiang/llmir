#!/usr/bin/env bash
# Fast A-class walkthrough for reviewers (~CPU, no GPU). Full regen: reproduce_paper.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

step() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

step "1/8 — E1 compile-time pass (pytest)"
pytest tests/test_mvp_a_e2e.py -m "not network" -q

step "2/8 — E2 shared-prefix decoder + S1/S2/S3 buckets"
pytest tests/test_sharegpt_prefix_bench.py tests/test_decoder_workload_buckets.py -m "not network" -q
python3 scripts/regenerate_decoder_workload_buckets.py --verify-only

step "3/8 — E3 GPU KV integration (CPU path)"
pytest tests/test_mvp_c_e2e.py tests/test_torch_gpu_kv_cache.py -m "not network" -q

step "4/8 — E4/E5 analytical harness (S2 + multi-bucket)"
pytest tests/test_e4_compositional.py tests/test_e5_ablation.py tests/test_e4_e5_multi_bucket.py -q

step "5/8 — E6 backend parity + M5 lowered hot path"
pytest tests/test_e6_backend_parity.py tests/test_m5_lowered_hot_path.py -q

step "6/9 — MLIR lit catalog (skipped without mlir-opt)"
pytest tests/test_mlir_lit_suite.py -q
bash scripts/mlir_lit_smoke.sh

step "7/9 — PyPI release alignment (optional network)"
python3 scripts/verify_pypi_release.py || true

step "8/9 — Optional E8 empirical GPU (B-class)"
bash scripts/e8_lab_smoke.sh

step "9/9 — M6 artifact bundle verify"
python3 scripts/verify_artifact_bundle.py --skip-figures

python3 scripts/walkthrough_summary.py
python3 scripts/generate_evidence_dashboard.py

echo ""
echo "Walkthrough complete."
echo "  Summary: IEEE-conference/benchmarks/walkthrough_summary.json"
echo "  Status: IEEE-conference/benchmarks/artifact_bundle_status.json"
echo "  Full CPU regen: bash scripts/reproduce_paper.sh"
echo "  Paper tables:   python3 scripts/generate_paper_bucket_tables_tex.py"
