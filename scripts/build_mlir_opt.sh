#!/usr/bin/env bash
# Build mlir-opt from the in-tree MLIR tree (LLM dialect + passes for lit suite).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-native}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "LLMIR mlir-opt build"
echo "  source: ${ROOT}"
echo "  build:  ${BUILD_DIR}"

bash "$(dirname "$0")/check_native_build_prereqs.sh" || true

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found; install cmake to build mlir-opt." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
CMAKE_ARGS=(
  -S "${ROOT}"
  -B "${BUILD_DIR}"
  -DLLVM_ENABLE_PROJECTS=mlir
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
  -DCMAKE_BUILD_TYPE=Release
  -DLLMIR_ENABLE_CUDA=OFF
)

if [[ ! -f "${BUILD_DIR}/build.ninja" && ! -f "${BUILD_DIR}/Makefile" ]]; then
  echo "Configuring CMake (first run may take several minutes)..."
  cmake "${CMAKE_ARGS[@]}"
fi

cmake --build "${BUILD_DIR}" --target mlir-opt -j"${JOBS}"

OPT_BIN="${BUILD_DIR}/bin/mlir-opt"
if [[ ! -x "${OPT_BIN}" ]]; then
  OPT_BIN="$(find "${BUILD_DIR}" -name mlir-opt -type f -executable | head -1)"
fi
if [[ -z "${OPT_BIN}" || ! -x "${OPT_BIN}" ]]; then
  echo "Build finished but mlir-opt not found under ${BUILD_DIR}" >&2
  exit 1
fi

echo ""
echo "Built: ${OPT_BIN}"
echo "export PATH=$(dirname "${OPT_BIN}"):\$PATH"
echo "export LLMIR_OPT_EXECUTABLE=${OPT_BIN}"
echo "# Then: bash scripts/mlir_lit_smoke.sh"
