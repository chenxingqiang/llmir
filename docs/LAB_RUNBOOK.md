# Lab Runbook (optional B-class + release checks)

CPU CI honestly records `skipped` for mlir lit and E8 GPU. Use this index for
**lab closure** on machines with LLVM, CUDA, or PyPI maintainer access.

## One-shot smoke (CPU or GPU)

```bash
bash scripts/lab_smoke_all.sh
python3 scripts/verify_lab_gates.py
cat IEEE-conference/benchmarks/lab_status_summary.json
```

## By concern

| Concern | Script | Workflow (dispatch) | Doc |
|---------|--------|---------------------|-----|
| MLIR lit (4 files) | `mlir_lit_smoke.sh` | MLIR lit lab (optional) | `MLIR_LIT_RUNBOOK.md` |
| E8 GPU bench | `e8_lab_smoke.sh` / `e8_lab_run.sh` | E8 GPU lab (optional) | `E8_LAB_RUNBOOK.md` |
| PyPI publish | `verify_pypi_release.py` | PyPI republish (maintainer) | `PYPI_TRUSTED_PUBLISHER.md` |
| Native build | `check_native_build_prereqs.sh` | Native Runtime (optional) | `MLIR_NATIVE_BUILD.md` |
| All optional | `lab_smoke_all.sh` | Lab smoke (optional) | this file |

## Strict vs smoke

| Mode | mlir lit | E8 GPU | Exit on skip |
|------|----------|--------|--------------|
| **Smoke** | `mlir_lit_smoke.sh` | `e8_lab_smoke.sh` | No (exit 0) |
| **Strict** | opt on PATH + `passed` | `e8_lab_run.sh` | Yes on failure |
| **MLIR strict** | `mlir_lit_preflight.sh --strict` + `require_passed` | MLIR lit lab | Yes on failure |
| **E8 strict** | `e8_lab_preflight.sh --strict` + `require_completed` | E8 GPU lab | Yes on failure |

E8 GPU lab workflow: set `require_completed=true` on a CUDA runner.

MLIR lit lab workflow: pass `mlir_opt_executable` from `build_mlir_opt.sh`; set `require_passed=true` for 4/4 green.

## Gates in A-class CI

- `walkthrough_a_class.sh` → `verify_walkthrough_gates.py` (includes `verify_lab_gates.py`)
- `prepare_release.sh` runs lab summary + gates before wheel build

## Expected CPU CI snapshot

```json
{
  "mlir_lit_status": "skipped",
  "e8_status": "skipped",
  "pypi_release_status": "pending",
  "native_build_prereqs_ok": false
}
```

This is **honest Tier-A** per `CAPABILITY_MATRIX.md`; lab workflows close gaps without blocking main.
