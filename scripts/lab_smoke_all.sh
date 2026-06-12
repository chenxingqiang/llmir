#!/usr/bin/env bash
# Run optional lab smokes (mlir lit, E8 GPU, PyPI align) — honest skip on CPU CI.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

echo "LLMIR lab smoke (all optional B-class / lab checks)"
echo ""

echo "=== Lab preflights (non-strict) ==="
bash scripts/mlir_lit_preflight.sh
bash scripts/e8_lab_preflight.sh

echo ""
echo "=== MLIR lit ==="
bash scripts/mlir_lit_smoke.sh

echo ""
echo "=== E8 empirical GPU ==="
bash scripts/e8_lab_smoke.sh

echo ""
echo "=== PyPI release alignment ==="
python3 scripts/verify_pypi_release.py || python3 scripts/verify_pypi_release.py --offline

echo ""
echo "=== Native build prerequisites ==="
bash scripts/check_native_build_prereqs.sh || true

echo ""
python3 scripts/lab_status_summary.py

echo ""
echo "=== Lab gates ==="
python3 scripts/verify_lab_gates.py

echo ""
echo "Lab smoke all complete."
echo "  Summary: IEEE-conference/benchmarks/lab_status_summary.json"
