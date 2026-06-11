# E6: Multi-Backend Correctness Parity

Paper **E6** verifies that **numpy** and **torch_cuda** KV backends produce **identical greedy decode tokens** and matching KV lookup tensors on the same workload. Performance is reported separately (MVP-C / optional E8), not mixed into E6.

## Commands

```bash
# Offline toy model (CPU, no network)
python3 scripts/e6_backend_parity_verify.py --model toy

pytest tests/test_e6_backend_parity.py -q
```

## Output

`IEEE-conference/benchmarks/e6_backend_parity.json`

| Section | Meaning |
|---------|---------|
| `decode_parity` | Per-prompt generated token ids per backend vs `numpy` reference |
| `kv_micro_parity` | Append/lookup numeric diff after identical KV tensors |
| `summary.overall_pass` | Both decode and KV micro checks pass |

## Optional backends

`native` may be appended when `libMLIRLLMRuntime` is built; failures are recorded per-row without failing the harness import.

## Paper wording

- **May claim:** CPU/GPU-resident KV paths agree on tokens and stored KV layout for the tested harness.
- **May not claim:** E6 proves throughput parity or beats vLLM (see E8).
