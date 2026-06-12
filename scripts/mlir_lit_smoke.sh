#!/usr/bin/env bash
# Run Tier-A MLIR lit catalog; writes status JSON (skipped without mlir-opt).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${LLMIR_OPT_EXECUTABLE:-}" && -z "${MLIR_OPT_EXECUTABLE:-}" ]]; then
  if ! command -v mlir-opt >/dev/null 2>&1 && ! command -v llmir-opt >/dev/null 2>&1; then
    echo "mlir-opt not found — lit suite will record status=skipped" >&2
    echo "Build: bash scripts/build_mlir_opt.sh (see docs/MLIR_LIT_RUNBOOK.md)" >&2
  fi
fi

python3 scripts/verify_mlir_lit_suite.py
