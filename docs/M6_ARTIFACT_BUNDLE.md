# M6: CPU Artifact Bundle

Engineering milestone **M6** packages **E1–E6 + M5** into a single reproducible, verifiable artifact set for reviewers and CI (no GPU).

## One-shot reproduce

```bash
bash scripts/reproduce_paper.sh
```

## Verify without full regen

```bash
python3 scripts/verify_artifact_bundle.py
pytest tests/test_artifact_bundle.py -q
```

Skip figure PDF requirement (when matplotlib regen not run):

```bash
python3 scripts/verify_artifact_bundle.py --skip-figures
```

## Manifest

`IEEE-conference/benchmarks/artifact_manifest.json` lists:

| ID | Artifact |
|----|----------|
| e1 | `gpt2_e1_snippet.mlir` |
| e2 | `shared_prefix_decoder_2048_sim.json` |
| e4–e6 | `e4_compositional.json`, `e5_ablation.json`, `e6_backend_parity.json` |
| m5 | `m5_lowered_hot_path.json` |
| e8 (optional) | `e8_empirical_gpu.json` — B-class; `skipped` on CPU OK |
| e4/e5 buckets (optional) | `e4_compositional_buckets.json`, `e5_ablation_buckets.json` |

Fast reviewer path (verify only, no full regen): `bash scripts/walkthrough_a_class.sh` — see `docs/WALKTHROUGH.md`.

Status output: `IEEE-conference/benchmarks/artifact_bundle_status.json`

## Paper wording

- **May claim:** artifacts are listed, scriptable, and CI-checkable on CPU.
- **May not claim:** bundle includes 7B GPU measured vs vLLM (see optional E8).
