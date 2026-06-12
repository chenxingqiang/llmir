# CI Workflow Index

Dispatch and automatic workflows for reviewers and maintainers.

## Automatic (on push/PR to `main`)

| Workflow | Purpose |
|----------|---------|
| [A-class walkthrough](https://github.com/chenxingqiang/llmir/actions/workflows/a-class-walkthrough.yml) | E1–E6 + M6 walkthrough + lab/walkthrough gates |
| [Python package](https://github.com/chenxingqiang/llmir/actions/workflows/python-package.yml) | pytest matrix, lint, build; **publish** on `v*` tags |
| [Native Runtime (optional)](https://github.com/chenxingqiang/llmir/actions/workflows/native-runtime.yml) | Best-effort prereq check + `libMLIRLLMRuntime` build (also dispatchable) |

## Manual (`workflow_dispatch`)

| Workflow | When to use |
|----------|-------------|
| [Native Runtime (optional)](https://github.com/chenxingqiang/llmir/actions/workflows/native-runtime.yml) | Prereq report + `libMLIRLLMRuntime` build; set `strict_prereqs` on LLVM runners |
| [Lab smoke (optional)](https://github.com/chenxingqiang/llmir/actions/workflows/lab-smoke.yml) | mlir/E8/PyPI/native rollup on CPU |
| [MLIR lit lab (optional)](https://github.com/chenxingqiang/llmir/actions/workflows/mlir-lit-lab.yml) | Preflight + `require_passed` for 4/4 lit green |
| [E8 GPU lab (optional)](https://github.com/chenxingqiang/llmir/actions/workflows/e8-gpu-lab.yml) | Preflight + `require_completed=true` on CUDA runner |
| [Release prep](https://github.com/chenxingqiang/llmir/actions/workflows/release-prep.yml) | Wheel build + lab JSON before tag |
| [Release tag (PyPI trigger)](https://github.com/chenxingqiang/llmir/actions/workflows/release-tag.yml) | Create/push annotated tag |
| [PyPI republish (maintainer)](https://github.com/chenxingqiang/llmir/actions/workflows/pypi-republish.yml) | Preflight + retry publish + `--require-published` verify |
| [E8 empirical GPU (legacy)](https://github.com/chenxingqiang/llmir/actions/workflows/e8-empirical-gpu.yml) | Same as E8 smoke on `ubuntu-latest` |
| [GPU inference compare (optional)](https://github.com/chenxingqiang/llmir/actions/workflows/gpu-benchmark.yml) | HF vs llmir-paged compare (GPU runner) |

## Local equivalents

```bash
bash scripts/walkthrough_a_class.sh          # A-class (CI walkthrough job)
bash scripts/lab_smoke_all.sh                # Lab smoke job
bash scripts/prepare_release.sh              # Release prep job
bash scripts/tag_release.sh --dry-run        # Release tag dry-run
bash scripts/check_native_build_prereqs.sh   # Native runtime prereq report
```

See also: [`LAB_RUNBOOK.md`](LAB_RUNBOOK.md), [`WALKTHROUGH.md`](WALKTHROUGH.md), [`PYPI_RELEASE_CHECKLIST.md`](PYPI_RELEASE_CHECKLIST.md).
