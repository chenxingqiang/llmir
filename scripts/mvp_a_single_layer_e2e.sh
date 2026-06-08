#!/usr/bin/env bash
# MVP-A: paper-aligned single-layer KV pipeline smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "=== pytest (MVP-A) ==="
python3 -m pytest tests/test_mvp_a_e2e.py -q

echo "=== llmir-compile MVP-A ==="
llmir-compile --mvp-a-e2e --run-reference --compare-torch \
  --seq-len 8 --num-heads 2 --head-dim 16 \
  --mvp-json /tmp/mvp_a_summary.json \
  -o /tmp/mvp_a_single_layer.mlir

if command -v mlir-opt >/dev/null || command -v llmir-opt >/dev/null; then
  OPT="$(command -v llmir-opt || command -v mlir-opt)"
  echo "=== mlir lit smoke (${OPT}) ==="
  "${OPT}" test/Dialect/LLM/mvp_single_layer_pipeline.mlir -llm-optimize-kv-cache \
    | head -n 20
fi

echo "MVP-A complete."
