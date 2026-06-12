#!/usr/bin/env bash
# Validate tag/version alignment before PyPI republish workflow_dispatch.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "Usage: pypi_republish_preflight.sh TAG" >&2
  echo "Example: pypi_republish_preflight.sh v0.2.2" >&2
  exit 2
fi

if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: invalid tag format (expected vX.Y.Z): $TAG" >&2
  exit 1
fi

VERSION="${TAG#v}"
echo "PyPI republish preflight: ${TAG}"
echo "=================================="

python3 scripts/pyproject_tools.py check-alignment "${VERSION}"

echo ""
echo "Trusted publisher must include:"
echo "  workflow: python-package.yml and/or pypi-republish.yml"
echo "  environment: pypi"
echo "See docs/PYPI_TRUSTED_PUBLISHER.md"
echo ""
echo "Preflight OK for ${TAG}"
