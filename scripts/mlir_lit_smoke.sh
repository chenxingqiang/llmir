#!/usr/bin/env bash
# Run Tier-A MLIR lit catalog when mlir-opt is on PATH (see docs/MLIR_LIT_RUNBOOK.md).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${LLMIR_OPT_EXECUTABLE:-}" && -z "${MLIR_OPT_EXECUTABLE:-}" ]]; then
  if ! command -v mlir-opt >/dev/null 2>&1 && ! command -v llmir-opt >/dev/null 2>&1; then
    echo "mlir-opt not found. Build with: bash scripts/build_mlir_opt.sh" >&2
    echo "See docs/MLIR_LIT_RUNBOOK.md" >&2
    exit 2
  fi
fi

python3 scripts/verify_mlir_lit_suite.py
