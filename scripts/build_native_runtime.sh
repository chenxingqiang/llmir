#!/usr/bin/env bash
# Build libMLIRLLMRuntime from the MLIR/LLVM tree (optional native KV + CUDA path).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-native}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
ENABLE_CUDA="${LLMIR_ENABLE_CUDA:-auto}"

echo "LLMIR native runtime build"
echo "  source: ${ROOT}"
echo "  build:  ${BUILD_DIR}"

bash "$(dirname "$0")/check_native_build_prereqs.sh" || true

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found; install cmake to build the native runtime." >&2
  exit 1
fi

CUDA_FLAG=OFF
if [[ "${ENABLE_CUDA}" == "ON" || "${ENABLE_CUDA}" == "1" ]]; then
  CUDA_FLAG=ON
elif [[ "${ENABLE_CUDA}" == "auto" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_FLAG=ON
    echo "  nvcc found: enabling LLMIR_ENABLE_CUDA"
  else
    echo "  nvcc not found: building CPU-only runtime"
  fi
fi

mkdir -p "${BUILD_DIR}"
CMAKE_ARGS=(
  -S "${ROOT}"
  -B "${BUILD_DIR}"
  -DLLVM_ENABLE_PROJECTS=mlir
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
  -DCMAKE_BUILD_TYPE=Release
  -DLLMIR_ENABLE_CUDA="${CUDA_FLAG}"
)

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --target MLIRLLMRuntime -j"${JOBS}"

SO="$(find "${BUILD_DIR}" -name 'libMLIRLLMRuntime.so' -o -name 'libMLIRLLMRuntime.dylib' | head -1)"
if [[ -z "${SO}" ]]; then
  echo "Build finished but libMLIRLLMRuntime shared library not found under ${BUILD_DIR}" >&2
  exit 1
fi

echo ""
echo "Built: ${SO}"
echo "export LLMIR_LIB_PATH=${SO}"
if [[ "${CUDA_FLAG}" == "ON" ]]; then
  echo "# CUDA kernels linked (CUDAKernels.cu)"
fi
