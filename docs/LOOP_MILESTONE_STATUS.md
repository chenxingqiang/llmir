# Loop Milestone Status (living)

Engineering milestones **M1–M8** and recent **Loop** iterations. Update after each merged optimization PR.

## Milestones

| ID | Name | Status |
|----|------|--------|
| M1 | E1 single-layer IR + CI | done |
| M2 | E4 compositional trace | done |
| M3 | E5 ablation switches | done |
| M4 | E6 multi-backend parity | done |
| M5 | Lowered hot path semantics | done |
| M6 | CPU artifact bundle | done |
| M7 | E8 scaffold + S1/S3 buckets + reproduce E8 | scaffold |
| M8 | MLIR lit catalog + S1/S2/S3 block-size lit | done |
| M9 | PyPI release prep (`prepare_release.sh`) | done |
| M10 | Release 0.2.1 version bump + MLIR lit runbook | done |
| M11 | Release tag automation (`tag_release.sh` + workflow) | done |
| M12 | PyPI release verify + trusted publisher runbook | done |
| M13 | Walkthrough PyPI step + mlir-lit-lab workflow | done |
| M14 | E8 GPU lab smoke + e8-gpu-lab workflow | done |
| M15 | Lab smoke hub + MLIR native build guide | done |
| M16 | Lab status wired into walkthrough + dashboard | done |
| M17 | Lab gates + PyPI republish workflow | done |
| M18 | Lab smoke CI + LAB_RUNBOOK hub | done |
| M19 | Reproduce paper lab tail alignment | done |
| M20 | CI explicit lab gates + release-prep artifacts | done |
| M21 | CI workflow index doc | done |
| M22 | Paper traceability lab/release commands | done |
| M23 | Native runtime CI prereq check + dispatch | done |
| M24 | PyPI republish preflight + post-publish verify | done |
| M25 | MLIR lit lab preflight + require_passed gates | done |

## Recent loops (R6–R13)

| Loop | Theme |
|------|--------|
| R6 | Paper §5 E4–E6/M6 + E8 scaffold |
| R7 | S1/S3 workload buckets + reproduce E8 |
| R8 | E4/E5 multi-bucket traces |
| R9 | Paper appendix bucket tables (TeX from JSON) |
| R10 | MLIR lit catalog + `decoder_workload_buckets.mlir` |
| R11 | `walkthrough_a_class.sh` + optional manifest entries |
| R12 | CI walkthrough workflow + E8 lab runbook |
| R13 | Walkthrough summary JSON + milestone doc |
| R14 | README CI badges + `EVIDENCE_DASHBOARD.md` |
| R15 | CI lint hardening + `verify_walkthrough_gates.py` |
| R16 | Python 3.9+ policy; CI matrix drops 3.8 |
| R17 | PyPI release prep script + checklist + workflow |
| R18 | MVP-C CUDA device wiring test (CPU-safe CI) |
| R19 | Release 0.2.1 + MLIR lit runbook (`build_mlir_opt.sh`) |
| R20 | Release tag automation + PyPI trigger workflow |
| R21 | CI release fixes (tomli, mypy) + 0.2.2 bump |
| R22 | PyPI verify JSON + trusted publisher docs |
| R23 | Walkthrough PyPI + mlir-lit-lab CI workflow |
| R24 | E8 GPU lab smoke + e8-gpu-lab workflow |
| R25 | Lab smoke hub + native build prerequisites |
| R26 | Lab status in evidence dashboard + walkthrough |
| R27 | verify_lab_gates + pypi-republish workflow |
| R28 | lab-smoke.yml + LAB_RUNBOOK hub |
| R29 | reproduce_paper.sh lab tail |
| R30 | CI lab gates step + release-prep lab artifacts |
| R31 | CI_WORKFLOW_INDEX.md |
| R32 | Paper traceability lab commands |
| R33 | Native runtime prereqs + workflow_dispatch |
| R34 | PyPI republish preflight + verify gates |
| R35 | MLIR lit lab preflight + require_passed |

## Reviewer commands

```bash
bash scripts/walkthrough_a_class.sh
python3 scripts/walkthrough_summary.py
cat IEEE-conference/benchmarks/walkthrough_summary.json
```

CI: `.github/workflows/a-class-walkthrough.yml`
