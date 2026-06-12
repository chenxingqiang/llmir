#!/usr/bin/env bash
# Validate MLIR lit catalog and optional mlir-opt before lab workflow_dispatch.
set -euo pipefail
cd "$(dirname "$0")/.."

STRICT=0
OPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    -h | --help)
      echo "Usage: mlir_lit_preflight.sh [--strict] [MLIR_OPT_PATH]"
      exit 0
      ;;
    *)
      OPT="$1"
      shift
      ;;
  esac
done

echo "MLIR lit preflight"
echo "=================="

for f in \
  kv_cache_ops.mlir \
  kv_cache_optimization.mlir \
  mvp_single_layer_pipeline.mlir \
  decoder_workload_buckets.mlir
do
  if [[ ! -f "test/Dialect/LLM/${f}" ]]; then
    echo "ERROR: missing catalog file ${f}" >&2
    exit 1
  fi
done
echo "  OK   lit catalog (4 Tier-A files)"

bash scripts/check_native_build_prereqs.sh || true

resolved=""
if [[ -n "$OPT" ]]; then
  resolved="$OPT"
elif [[ -n "${LLMIR_OPT_EXECUTABLE:-}" ]]; then
  resolved="${LLMIR_OPT_EXECUTABLE}"
elif [[ -n "${MLIR_OPT_EXECUTABLE:-}" ]]; then
  resolved="${MLIR_OPT_EXECUTABLE}"
elif command -v mlir-opt >/dev/null 2>&1; then
  resolved="$(command -v mlir-opt)"
elif command -v llmir-opt >/dev/null 2>&1; then
  resolved="$(command -v llmir-opt)"
fi

if [[ -n "$resolved" ]]; then
  if [[ ! -x "$resolved" ]]; then
    echo "ERROR: mlir-opt not executable: ${resolved}" >&2
    exit 1
  fi
  if ! "$resolved" --help >/dev/null 2>&1; then
    echo "ERROR: mlir-opt failed --help: ${resolved}" >&2
    exit 1
  fi
  echo "  OK   mlir-opt: ${resolved}"
else
  echo "  MISS mlir-opt (build: bash scripts/build_mlir_opt.sh)"
  if [[ "$STRICT" == "1" ]]; then
    echo "ERROR: --strict requires in-tree mlir-opt on lab runner" >&2
    exit 1
  fi
fi

echo ""
echo "Next:"
echo "  bash scripts/build_mlir_opt.sh"
echo "  export LLMIR_OPT_EXECUTABLE=\${PWD}/build-native/bin/mlir-opt"
echo "  bash scripts/mlir_lit_smoke.sh"
echo "  python3 scripts/verify_mlir_lit_suite.py --require-passed"
echo ""
echo "Preflight OK"
