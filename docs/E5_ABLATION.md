# E5: Ablation at Verifiable Layers

Paper **E5** toggles **E1 block-opt**, **E2 prefix cache**, and **E3 GPU-resident KV** independently on a shared-prefix decoder trace. Each switch changes only the expected analytical proxy (CPU/CI reproducible).

## Commands

```bash
# S2 bucket (2048-token prefix)
python3 scripts/e5_ablation_verify.py --from-sim \
  IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json

pytest tests/test_e5_ablation.py -q

# All S1/S2/S3 buckets
python3 scripts/e4_e5_multi_bucket_verify.py --e5-only
pytest tests/test_e4_e5_multi_bucket.py -q
```

## Output

`IEEE-conference/benchmarks/e5_ablation.json` (S2 primary artifact for M6)

`IEEE-conference/benchmarks/e5_ablation_buckets.json` (S1/S2/S3 aggregate)

| Section | Meaning |
|---------|---------|
| `configurations` | Named switch rows: baseline, isolated (e1/e2/e3 only), cumulative stack, full |
| `proxies` | `block_size_reduction_ratio`, `prefill_reduction_ratio`, `host_copy_reduction_ratio`, … |
| `delta_vs_baseline` | Per-row delta from the all-off baseline |
| `isolated_contributions` | Single-knob deltas for E1 / E2 / E3 |
| `cumulative_stack` | baseline → +E1 → +E1+E2 → full |

## Switches

| Switch | On | Off |
|--------|----|-----|
| `block_opt` | E1 `optimize_block_size_attr` (1024→64) | Keep oversized `block_size` attr |
| `prefix_cache` | Warm shared-prefix prefill | Cold full prefill per request |
| `torch_cuda_kv` | 0 host round-trips on decode hot path | NumPy host-staging path |

## Paper wording

- **May claim:** each compile-time lever moves the predicted proxy when enabled in isolation or cumulatively.
- **May not claim:** tokens/sec from the illustrative `tab:ablation` table (projected appendix only).
