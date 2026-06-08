# E1: Compile-Time Pass Verification

Paper **E1** verifies that compile-time KV block optimization (Algorithm 1) is applied in IR and matches a numeric reference kernel.

Repository harness (legacy CLI): `llmir-compile --mvp-a-e2e`, `tests/test_mvp_a_e2e.py`, `IEEE-conference/benchmarks/gpt2_mvp_a_snippet.mlir`.

## Commands

```bash
pytest tests/test_mvp_a_e2e.py -m "not network" -q

llmir-compile --mvp-a-e2e --run-opt --run-reference --compare-torch \
  --seq-len 8 --mvp-json /tmp/e1.json -o /tmp/e1.mlir

mlir-opt test/Dialect/LLM/mvp_single_layer_pipeline.mlir -llm-optimize-kv-cache
```

## Success criteria

- Block attribute reduced (e.g., 1024 → 32 for short sequences).
- Reference kernel numeric agreement within tolerance after `llm-optimize-kv-cache`.
