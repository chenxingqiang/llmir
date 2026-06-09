# E2: Prefix-Aware Serving Evaluation

Paper **E2** demonstrates **product value** from prefix KV reuse (paper §5.1 workload shape), complementary to E1's compile-time IR path.

Repository harness (legacy CLI): `llmir-benchmark --sharegpt-prefix-bench`, `scripts/sharegpt_prefix_bench.py`.

## Workload

One long **shared system prompt** + **N user-variant** suffixes (synthetic ShareGPT):

| Parameter | CI default | GPU demo |
|-----------|------------|----------|
| `system_prompt_tokens` | 128 | 2048 |
| `num_requests` | 32 | 32 |
| `user_suffix_tokens` | 8 | 8–32 |

## Commands

```bash
# KV-layer simulation only (no HuggingFace)
python scripts/sharegpt_prefix_bench.py --simulation-only

# Full E2 harness (sim + llmir_paged on gpt2)
llmir-benchmark --sharegpt-prefix-bench --model gpt2 \
  --sharegpt-system-tokens 128 --sharegpt-requests 32 -o sharegpt.json

# GPU-scale demo
python scripts/sharegpt_prefix_bench.py --model gpt2 --device cuda \
  --system-prompt-tokens 2048 --num-requests 32 -o sharegpt_gpu.json
```

## vLLM connector (P4)

Disk-backed prefix KV for vLLM V1 disaggregated prefill:

```bash
python scripts/vllm_kv_connector_smoke.py
python scripts/vllm_kv_connector_smoke.py --register  # when vLLM installed
```

See [VLLM_KV_CONNECTOR.md](./VLLM_KV_CONNECTOR.md).

## Artifacts

| File | Content |
|------|---------|
| `IEEE-conference/benchmarks/sharegpt_2048_sim.json` | 2048-token system prompt KV sim |
| `IEEE-conference/benchmarks/paper_results.json` | gpt2 `warm_prefix` prefill tokens |

## Success criteria

- **KV simulation**: `sharegpt_kv_prefix_cached` speedup ≫ 1 vs baseline (ideal prefix append).
- **llmir_paged E2E**: `sharegpt_llmir_warm_prefix` lower total time vs `sharegpt_llmir_no_prefix`; rising `prefix_hit_tokens` on requests after `warm_prefix`.

## Honesty

- Synthetic word-count prompts approximate token lengths; use a tokenizer for exact ShareGPT replay.
- This is **not** the paper Table II multi-model vLLM throughput table yet.
