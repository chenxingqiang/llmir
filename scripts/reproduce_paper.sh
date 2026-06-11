#!/usr/bin/env bash
# M6 CPU artifact bundle: reproduce E1–E6 + M5 and verify manifest.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"
ROOT="$PWD"

echo "LLMIR A-class artifact reproduce (CPU, no GPU)"
echo "manifest: IEEE-conference/benchmarks/artifact_manifest.json"
echo ""

echo "=== E1 compile pass ==="
pytest tests/test_mvp_a_e2e.py -m "not network" -q

echo "=== E2 prefix decoder ==="
pytest tests/test_sharegpt_prefix_bench.py -m "not network" -q

echo "=== E3 GPU KV integration ==="
pytest tests/test_mvp_c_e2e.py tests/test_torch_gpu_kv_cache.py -m "not network" -q

echo "=== E4 compositional ==="
pytest tests/test_e4_compositional.py -q
python3 scripts/e4_compositional_verify.py --from-sim \
  IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json

echo "=== E5 ablation ==="
pytest tests/test_e5_ablation.py -q
python3 scripts/e5_ablation_verify.py --from-sim \
  IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json

echo "=== E6 backend parity ==="
pytest tests/test_e6_backend_parity.py -q
python3 scripts/e6_backend_parity_verify.py --model toy

echo "=== M5 lowered hot path ==="
pytest tests/test_m5_lowered_hot_path.py -q
python3 scripts/m5_lowered_hot_path_verify.py

echo "=== Paper JSON (optional, needs llmir[full] for HF) ==="
if python3 -c "import transformers" 2>/dev/null; then
  python3 scripts/paper_benchmark_collect.py --model gpt2 --sharegpt-simulation-only \
    --prompt-tokens 8 --max-tokens 2 --warmup 0 || true
else
  echo "skip paper_benchmark_collect (install llmir[full] for HF)"
fi

echo "=== Figures ==="
if python3 -c "import matplotlib" 2>/dev/null; then
  python3 IEEE-conference/figures/generate_all_nature_figures.py
  FIGURE_CHECK=1
else
  echo "skip figures (pip install matplotlib)"
  FIGURE_CHECK=0
fi

echo "=== M6 artifact verify ==="
if [[ "${FIGURE_CHECK:-0}" == "1" ]]; then
  python3 scripts/verify_artifact_bundle.py
else
  python3 scripts/verify_artifact_bundle.py --skip-figures
fi

echo ""
echo "Done. Artifacts: IEEE-conference/benchmarks/"
echo "Status: IEEE-conference/benchmarks/artifact_bundle_status.json"
