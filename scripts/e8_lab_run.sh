#!/usr/bin/env bash
# E8 B-class lab runner: requires CUDA for status=completed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

OUT="IEEE-conference/benchmarks/e8_empirical_gpu.json"
MODEL="${E8_MODEL:-gpt2}"
BACKENDS="${E8_BACKENDS:-hf,llmir-paged}"

echo "E8 empirical GPU lab run (B-class)"
echo "model=$MODEL backends=$BACKENDS"
echo ""

python3 -c "
import torch
from llmir.runtime.cuda_probe import summarize_cuda_stack
stack = summarize_cuda_stack()
print('cuda_stack:', stack)
if not stack.get('torch_cuda'):
    raise SystemExit(
        'No CUDA available. E8 will write status=skipped (honest). '
        'Run on a GPU lab machine or workflow_dispatch e8-empirical-gpu.yml.'
    )
"

pytest tests/test_e8_empirical_gpu.py -q
python3 scripts/e8_empirical_gpu_bench.py --model "$MODEL" --backends "$BACKENDS" -o "$OUT"

python3 -c "
import json
from pathlib import Path
d = json.loads(Path('$OUT').read_text())
status = d.get('status')
rows = len(d.get('results', []))
print(f'E8 status={status} rows={rows}')
if status != 'completed':
    raise SystemExit('Expected status=completed on GPU lab; got ' + str(status))
if rows < 1:
    raise SystemExit('E8 completed but no result rows')
print('E8 lab run OK')
"
