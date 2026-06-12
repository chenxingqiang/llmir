#!/usr/bin/env bash
# Validate E8 GPU lab prerequisites before workflow_dispatch strict mode.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

STRICT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    -h | --help)
      echo "Usage: e8_lab_preflight.sh [--strict]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

MODEL="${E8_MODEL:-gpt2}"
BACKENDS="${E8_BACKENDS:-hf,llmir-paged}"

echo "E8 GPU lab preflight"
echo "=================="
echo "model=${MODEL} backends=${BACKENDS}"

for script in e8_lab_smoke.sh e8_lab_run.sh e8_empirical_gpu_bench.py verify_e8_lab.py; do
  if [[ ! -f "scripts/${script}" ]]; then
    echo "ERROR: missing scripts/${script}" >&2
    exit 1
  fi
done
echo "  OK   E8 lab scripts"

python3 - <<PY
import sys

strict = ${STRICT}

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError as exc:
    print(f"  MISS Python deps: {exc}")
    if strict:
        sys.exit(1)
else:
    print("  OK   torch + transformers import")

from llmir.runtime.cuda_probe import summarize_cuda_stack

stack = summarize_cuda_stack()
print(f"cuda_stack: {stack}")
if stack.get("torch_cuda"):
    print(f"  OK   torch CUDA ({stack.get('device_count', 0)} device(s))")
else:
    print("  MISS CUDA (honest skip on CPU; use --strict on GPU lab runner)")
    if strict:
        sys.exit(1)

if strict and not stack.get("torch_cuda"):
    sys.exit(1)
PY

echo ""
echo "Next:"
echo "  bash scripts/e8_lab_smoke.sh              # honest skip OK"
echo "  bash scripts/e8_lab_run.sh              # strict GPU lab"
echo "  python3 scripts/verify_e8_lab.py --require-completed"
echo ""
echo "Preflight OK"
