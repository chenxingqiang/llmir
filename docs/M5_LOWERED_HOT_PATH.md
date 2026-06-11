# M5: Lowered Op Hot Path

Engineering milestone **M5** closes the gap between **`-llm-lower-kv-cache-ops`** and a **verifiable execution path** for the single-layer KV micro-pipeline.

## What M5 proves

1. **Lowering** (when `mlir-opt` is available): high-level `llm.append_kv` / `llm.lookup_kv` / `llm.paged_attention` lower to `mlir_llm_*` runtime calls.
2. **Semantic hot path**: Python executes the same **append → lookup → paged_attention** order as lowered IR, using `create_paged_kv_cache` + reference attention.
3. **Parity**: hot-path output matches `compile_kv_micro_pipeline` reference within tolerance.

## Commands

```bash
pytest tests/test_m5_lowered_hot_path.py -q
python3 scripts/m5_lowered_hot_path_verify.py
```

## Output

`IEEE-conference/benchmarks/m5_lowered_hot_path.json`

| Field | Meaning |
|-------|---------|
| `mlir_lowered` | All `mlir_llm_*` symbols present in lowered MLIR |
| `execution_path` | e.g. `semantic_lowered::numpy` |
| `matches_reference` | Numeric parity vs reference pipeline |

## Scope boundary

- **Not** full `llmir_paged` decode hot path (HF model loop) — that remains integration via E3/E6.
- **`mlir_llm_paged_attention` C++** is still a stub; attention uses `numpy_paged_attention` on looked-up KV until native kernel lands.
- **Not** operator-level FlashAttention claims — see `benchmark/attention/` appendix only.
