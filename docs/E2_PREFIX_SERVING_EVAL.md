# E2: Prefix-Aware Serving Evaluation

Paper **E2** evaluates **shared-prefix decoder prefill** (RAG / multi-tenant): one long shared context + N suffix variants. This matches Qwen/Gemma/DeepSeek decoder serving patterns, **not** the ShareGPT dataset brand.

See [`DECODER_WORKLOAD_ARCHITECTURES.md`](./DECODER_WORKLOAD_ARCHITECTURES.md) for architecture presets and length buckets (S1/S2/S3).

Repository harness (legacy script/flags retained): `llmir-benchmark --shared-prefix-bench`, `scripts/sharegpt_prefix_bench.py`.

## Workload shapes

| Bucket | Shared prefix \(L_s\) | Suffix \(L_u\) | Requests \(N\) | Architecture target |
|--------|----------------------|----------------|----------------|---------------------|
| **S1** short | 128 | 8–32 | 32 | Qwen3 / Gemma 3 |
| **S2** RAG | 2048 | 8–32 | 32 | Qwen3-8B, Gemma-3-12B, DeepSeek-V3 |
| **S3** long-doc | 8192 | 64+ | 16 | Gemma 3 / Qwen2.5-72B |

gpt2 runs are **integration smoke only** (MHA + GELU-FFN, not representative of mainline decoders).

## Commands

```bash
# KV-layer simulation only (no HuggingFace)
python scripts/sharegpt_prefix_bench.py --simulation-only

# E2 on architecture-representative preset (needs llmir[full])
llmir-benchmark --shared-prefix-bench --model qwen3-8b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32 -o e2_qwen3.json

llmir-benchmark --shared-prefix-bench --model gemma-3-12b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32 -o e2_gemma3.json

# CI smoke
llmir-benchmark --shared-prefix-bench --model gpt2 \
  --shared-prefix-tokens 128 --shared-prefix-requests 32 -o e2_gpt2.json
```

## Artifacts

| File | Content |
|------|---------|
| `IEEE-conference/benchmarks/shared_prefix_decoder_128_sim.json` | **S1** bucket (128-token shared prefix KV sim) |
| `IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json` | **S2** bucket (2048-token shared prefix KV sim) |
| `IEEE-conference/benchmarks/shared_prefix_decoder_8192_sim.json` | **S3** bucket (8192-token shared prefix KV sim) |
| `IEEE-conference/benchmarks/paper_results.json` | gpt2 `warm_prefix` prefill tokens |

Regenerate or verify all buckets:

```bash
python3 scripts/regenerate_decoder_workload_buckets.py --verify-only
python3 scripts/regenerate_decoder_workload_buckets.py   # rewrite JSON
```

## Success criteria

- **KV simulation**: prefix-cached row speedup ≫ 1 vs baseline.
- **llmir_paged E2E**: warm-prefix row lower latency vs cold; rising `prefix_hit_tokens`.

## Honesty

- Lengths are tokenizer-accurate when `--model` uses HF tokenizer; word-count fallback is approximate.
- Not a multi-model vLLM throughput table; see E8 (optional empirical) in `PAPER_TOP_TIER_BAR.md`.
