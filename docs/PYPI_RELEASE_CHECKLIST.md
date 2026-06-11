# PyPI Release Checklist (0.2.x)

Use before tagging `v*` on `main`. Publishing uses `.github/workflows/python-package.yml` (`publish` job on tag).

## Pre-flight (local or CI)

```bash
bash scripts/prepare_release.sh
```

Or GitHub Actions → **Release prep** → `workflow_dispatch`.

## Maintainer steps

1. Move `[Unreleased]` entries in `CHANGELOG.md` to `[X.Y.Z] - YYYY-MM-DD`.
2. Bump `version` in `pyproject.toml` to match the tag (e.g. `0.2.1`).
3. Commit: `chore: release X.Y.Z`
4. Tag and push:
   ```bash
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   ```
5. Confirm **Python package** workflow: `test`, `lint`, `build`, `publish` all green.
6. Verify on PyPI: https://pypi.org/project/llmir/

## Evidence bar (A-class)

| Gate | Command |
|------|---------|
| Walkthrough | `bash scripts/walkthrough_a_class.sh` |
| Lint | `bash scripts/ci_lint_gate.sh` |
| Artifacts | `python3 scripts/verify_artifact_bundle.py --skip-figures` |

Optional B-class: `bash scripts/e8_lab_run.sh` on GPU lab (`status=completed`).

## Not in the wheel

- MLIR C++ tree / `mlir-opt` (build separately)
- `IEEE-conference/benchmarks/*.json` ship in git, not required on PyPI
- Native `libMLIRLLMRuntime.so` — see `docs/P5_CUDA_NATIVE.md`

## Version policy

Python **3.9+** only. See `docs/PYTHON_VERSION_POLICY.md`.
