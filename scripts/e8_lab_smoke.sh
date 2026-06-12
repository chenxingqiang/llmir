#!/usr/bin/env bash
# E8 B-class smoke: always writes e8_empirical_gpu.json; exit 0 when skipped or completed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="${HOME}/.local/bin:${PATH}"

OUT="IEEE-conference/benchmarks/e8_empirical_gpu.json"
MODEL="${E8_MODEL:-gpt2}"
BACKENDS="${E8_BACKENDS:-hf,llmir-paged}"

echo "E8 empirical GPU smoke (B-class)"
echo "model=$MODEL backends=$BACKENDS"
echo ""

pytest tests/test_e8_empirical_gpu.py -q
python3 scripts/e8_empirical_gpu_bench.py --model "$MODEL" --backends "$BACKENDS" -o "$OUT"

python3 - <<'PY'
import json
import sys
from pathlib import Path

out = Path("IEEE-conference/benchmarks/e8_empirical_gpu.json")
data = json.loads(out.read_text(encoding="utf-8"))
assert data.get("experiment") == "E8"
assert data.get("evidence_class") == "B"
status = data.get("status")
if status not in ("skipped", "completed"):
    raise SystemExit(f"unexpected E8 status: {status!r}")
print(f"E8 smoke OK: status={status}")
if status == "skipped":
    print("NOTE: no CUDA — honest skip (use e8_lab_run.sh on GPU lab for completed)")
PY
