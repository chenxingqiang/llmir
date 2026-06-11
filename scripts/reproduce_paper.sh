#!/usr/bin/env bash
# Reproduce A-class paper evidence (E1–E5) on CPU — no GPU required.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

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

echo "=== Paper JSON (optional, needs llmir[full] for HF) ==="
if python3 -c "import transformers" 2>/dev/null; then
  python3 scripts/paper_benchmark_collect.py --model gpt2 --sharegpt-simulation-only \
    --prompt-tokens 8 --max-tokens 2 --warmup 0 || true
else
  echo "skip paper_benchmark_collect (install llmir[full] for HF)"
fi

echo "=== Figures ==="
python3 IEEE-conference/figures/generate_all_nature_figures.py

echo "Done. Artifacts under IEEE-conference/benchmarks/ and figures/"
