# E8: Empirical GPU Benchmark (Optional, B-Class)

Paper **E8** is **B-class evidence**: same-harness throughput vs HF/vLLM on GPU. It is **not** required for Tier-A compiler claims and **cannot** be derived from IR correctness.

## Commands

```bash
# CPU VM: writes skipped status (honest)
python3 scripts/e8_empirical_gpu_bench.py

pytest tests/test_e8_empirical_gpu.py -q
```

## Output

`IEEE-conference/benchmarks/e8_empirical_gpu.json`

| Field | Meaning |
|-------|---------|
| `status` | `skipped` (no CUDA) or `completed` |
| `evidence_class` | Always `B` |
| `results` | `llmir-benchmark --compare` rows when GPU available |

## Paper wording

- **May claim (appendix / footnote):** empirical comparison on stated hardware SKU when `status=completed`.
- **May not claim:** E8 results prove compile-time theorems; do not mix with E4 compositional analysis.
