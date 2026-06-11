#!/usr/bin/env bash
# Mirror python-package.yml lint job locally.
set -euo pipefail
cd "$(dirname "$0")/.."
ruff check src/llmir
black --check src/llmir
mypy src/llmir/benchmark src/llmir/compiler --ignore-missing-imports
echo "ci_lint_gate: OK"
