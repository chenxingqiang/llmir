#!/usr/bin/env bash
# Copy a built libMLIRLLMRuntime into the Python package tree for local wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_SRC="${1:-${LLMIR_LIB_PATH:-}}"

if [[ -z "${LIB_SRC}" || ! -f "${LIB_SRC}" ]]; then
  echo "Usage: LLMIR_LIB_PATH=/path/to/libMLIRLLMRuntime.so $0" >&2
  echo "   or: $0 /path/to/libMLIRLLMRuntime.so" >&2
  exit 1
fi

DEST_DIR="${ROOT}/src/llmir/native"
mkdir -p "${DEST_DIR}"
BASENAME="$(basename "${LIB_SRC}")"
cp -f "${LIB_SRC}" "${DEST_DIR}/${BASENAME}"

cat > "${DEST_DIR}/README.txt" <<EOF
Bundled libMLIRLLMRuntime for local development wheels.
Set LLMIR_LIB_PATH to this file or install via pip after build.

Source: ${LIB_SRC}
EOF

echo "Copied native library to ${DEST_DIR}/${BASENAME}"
echo "pip install -e .  # then export LLMIR_LIB_PATH=${DEST_DIR}/${BASENAME}"
