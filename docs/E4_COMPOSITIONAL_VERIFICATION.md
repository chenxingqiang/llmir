# E4: Compositional / Trace-Driven Verification

Paper **E4** composes **E1 block sizing + E2 prefix prefill + E3 host-copy bounds** on a shared-prefix decoder trace \((L_s, N, L_u)\). This is **A-class** evidence (analytical, CI-reproducible), not end-to-end vs vLLM.

## Commands

```bash
# S2 bucket (2048-token prefix) + compare to E2 sim JSON
python3 scripts/e4_compositional_verify.py --from-sim \
  IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json

pytest tests/test_e4_compositional.py -q
```

## Output

`IEEE-conference/benchmarks/e4_compositional.json`

| Section | Meaning |
|---------|---------|
| `e1_block_sizing` | `block_size` attr before/after E1 pass + allocated-token accounting |
| `e2_prefix_prefill` | Cold vs warm prefill token accounting |
| `e3_kv_host_copies` | NumPy vs torch_cuda host round-trip upper bound |
| `composite` | Combined ratios + optional KV byte estimate from `ModelRegistry` |
| `measured_comparison` | E2 sim speedup vs analytical ideal bound |

## Paper wording

- **May claim:** compile-time levers shrink oversized `block_size` attrs, repeated prefill, and host copies under stated trace.
- **May not claim:** proved faster than vLLM on 7B GPU (see **E8** optional).
