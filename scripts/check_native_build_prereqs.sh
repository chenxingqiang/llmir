#!/usr/bin/env bash
# Report toolchain prerequisites for mlir-opt / libMLIRLLMRuntime builds.
set -euo pipefail

STRICT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1; shift ;;
    -h | --help)
      echo "Usage: check_native_build_prereqs.sh [--strict]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

missing=()

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    echo "  OK   $1 ($("$1" --version 2>/dev/null | head -1 || echo present))"
    return 0
  fi
  missing+=("$1")
  echo "  MISS $1"
  return 1
}

echo "Native / MLIR build prerequisites"
echo "=================================="

check_cmd cmake || true
check_cmd ninja || check_cmd make || missing+=("ninja-or-make")
check_cmd c++ || check_cmd clang++ || missing+=("c++")

llvm_ok=0
if command -v llvm-config >/dev/null 2>&1; then
  echo "  OK   llvm-config ($(llvm-config --version))"
  llvm_ok=1
elif [[ -n "${LLVM_DIR:-}" && -f "${LLVM_DIR}/LLVMConfig.cmake" ]]; then
  echo "  OK   LLVM_DIR=${LLVM_DIR}"
  llvm_ok=1
else
  echo "  MISS llvm-config or LLVM_DIR (LLVM development packages / monorepo build)"
  missing+=("llvm")
fi

if [[ "$llvm_ok" == "1" ]]; then
  echo ""
  echo "Next:"
  echo "  bash scripts/build_mlir_opt.sh"
  echo "  bash scripts/build_native_runtime.sh"
else
  echo ""
  echo "See docs/MLIR_NATIVE_BUILD.md for LLVM/MLIR setup."
fi

if ((${#missing[@]} > 0)); then
  echo ""
  echo "Missing: ${missing[*]}"
  if [[ "$STRICT" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

echo ""
echo "Prerequisites look sufficient to attempt an in-tree build."
exit 0
