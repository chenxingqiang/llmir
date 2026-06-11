#!/usr/bin/env bash
# Pre-release gate: lint + walkthrough smoke + wheel build (no PyPI upload).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

echo "LLMIR release prep (local)"
echo "version: $(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
echo ""

echo "=== Lint gate ==="
bash scripts/ci_lint_gate.sh

echo ""
echo "=== Fast pytest gates ==="
pytest tests/test_python_version_policy.py \
  tests/test_verify_walkthrough_gates.py \
  tests/test_artifact_bundle.py \
  tests/test_walkthrough_summary.py \
  -q

echo ""
echo "=== Walkthrough summary ==="
python3 scripts/walkthrough_summary.py
python3 scripts/generate_evidence_dashboard.py

echo ""
echo "=== Build wheel/sdist ==="
python3 -m pip install -q build twine
rm -rf dist/
python3 -m build
twine check dist/*

echo ""
echo "Release prep OK."
echo "  Next: update CHANGELOG, tag vX.Y.Z, push tag (publishes via python-package.yml)"
