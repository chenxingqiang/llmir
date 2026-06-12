# Lab Closure Matrix

Single index for **smoke** (CPU-honest) vs **strict** (lab machine) closure across
optional B-class and release checks. Tier-A CI uses smoke only; strict modes run via
`workflow_dispatch` on LLVM/CUDA/PyPI lab runners.

## Matrix

| Concern | Smoke (exit 0 on skip) | Strict (fail if gap) | Workflow | Doc |
|---------|------------------------|----------------------|----------|-----|
| MLIR lit (4 files) | `mlir_lit_smoke.sh` | `mlir_lit_preflight.sh --strict` + `verify_mlir_lit_suite.py --require-passed` | MLIR lit lab | `MLIR_LIT_RUNBOOK.md` |
| E8 GPU bench | `e8_lab_smoke.sh` | `e8_lab_preflight.sh --strict` + `verify_e8_lab.py --require-completed` | E8 GPU lab | `E8_LAB_RUNBOOK.md` |
| PyPI publish | `verify_pypi_release.py` | `pypi_republish_preflight.sh vX.Y.Z` + `--require-published` | PyPI republish | `PYPI_TRUSTED_PUBLISHER.md` |
| Native runtime | `check_native_build_prereqs.sh` | same + `native-runtime.yml strict_prereqs=true` | Native Runtime | `MLIR_NATIVE_BUILD.md` |
| All optional | `lab_smoke_all.sh` | per-row strict workflows above | Lab smoke | `LAB_RUNBOOK.md` |

## One-shot smoke (CPU CI equivalent)

```bash
bash scripts/lab_smoke_all.sh
python3 scripts/verify_lab_gates.py
cat IEEE-conference/benchmarks/lab_status_summary.json
```

`lab_smoke_all.sh` runs non-strict preflights, all smokes, native prereq report, summary, and lab gates.

## Strict lab sequence (examples)

**MLIR lit (LLVM runner)**

```bash
bash scripts/build_mlir_opt.sh
export LLMIR_OPT_EXECUTABLE="${PWD}/build-native/bin/mlir-opt"
bash scripts/mlir_lit_preflight.sh --strict "$LLMIR_OPT_EXECUTABLE"
bash scripts/mlir_lit_smoke.sh
python3 scripts/verify_mlir_lit_suite.py --require-passed
```

**E8 GPU (CUDA runner)**

```bash
bash scripts/e8_lab_preflight.sh --strict
bash scripts/e8_lab_run.sh
python3 scripts/verify_e8_lab.py --require-completed
```

**PyPI (maintainer)**

```bash
bash scripts/pypi_republish_preflight.sh v0.2.2
# GitHub Actions → PyPI republish → verify --require-published
```

## A-class walkthrough alignment

Local and CI walkthrough end with lab + walkthrough gates:

```bash
bash scripts/walkthrough_a_class.sh
python3 scripts/verify_walkthrough_gates.py
```

CI: `.github/workflows/a-class-walkthrough.yml` (includes explicit `verify_lab_gates` step).

## Honest CPU snapshot

```json
{
  "mlir_lit_status": "skipped",
  "e8_status": "skipped",
  "pypi_release_status": "pending",
  "native_build_prereqs_ok": false
}
```

This is expected Tier-A per `CAPABILITY_MATRIX.md`. Strict workflows close gaps without blocking `main`.
