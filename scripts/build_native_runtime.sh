#!/usr/bin/env bash
# Build libMLIRLLMRuntime from the MLIR/LLVM tree (optional native KV path).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-native}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "LLMIR native runtime build"
echo "  source: ${ROOT}"
echo "  build:  ${BUILD_DIR}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found; install cmake to build the native runtime." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" --target MLIRLLMRuntime -j"${JOBS}"

SO="$(find "${BUILD_DIR}" -name 'libMLIRLLMRuntime.so' -o -name 'libMLIRLLMRuntime.dylib' | head -1)"
if [[ -z "${SO}" ]]; then
  echo "Build finished but libMLIRLLMRuntime shared library not found under ${BUILD_DIR}" >&2
  exit 1
fi

echo ""
echo "Built: ${SO}"
echo "export LLMIR_LIB_PATH=${SO}"
