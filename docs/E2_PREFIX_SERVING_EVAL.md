# E2: Prefix-Aware Serving Evaluation

Paper **E2** evaluates **shared-prefix decoder prefill** (RAG / multi-tenant): one long shared context + N suffix variants. This matches Llama/Qwen-class serving patterns, **not** the ShareGPT dataset brand.

See [`DECODER_WORKLOAD_ARCHITECTURES.md`](./DECODER_WORKLOAD_ARCHITECTURES.md) for architecture presets and length buckets (S1/S2/S3).

Repository harness (legacy script/flags retained): `llmir-benchmark --shared-prefix-bench`, `scripts/sharegpt_prefix_bench.py`.

## Workload shapes

| Bucket | Shared prefix \(L_s\) | Suffix \(L_u\) | Requests \(N\) | Architecture target |
|--------|----------------------|----------------|----------------|---------------------|
| **S1** short | 128 | 8–32 | 32 | Llama-3 / Qwen2 (CI default) |
| **S2** RAG | 2048 | 8–32 | 32 | Llama-3-8B, Qwen2-7B, DeepSeek-7B |
| **S3** long-doc | 8192 | 64+ | 16 | Mistral / long-context |

gpt2 runs are **integration smoke only** (MHA + GELU-FFN, not representative of mainline decoders).

## Commands

```bash
# KV-layer simulation only (no HuggingFace)
python scripts/sharegpt_prefix_bench.py --simulation-only

# E2 on architecture-representative preset (needs llmir[full])
llmir-benchmark --shared-prefix-bench --model llama3-8b \
  --shared-prefix-tokens 2048 --shared-prefix-requests 32 -o e2_llama3.json

# CI smoke
llmir-benchmark --shared-prefix-bench --model gpt2 \
  --shared-prefix-tokens 128 --shared-prefix-requests 32 -o e2_gpt2.json
```

## Artifacts

| File | Content |
|------|---------|
| `IEEE-conference/benchmarks/shared_prefix_decoder_2048_sim.json` | S2 bucket, 2048-token shared prefix KV sim |
| `IEEE-conference/benchmarks/paper_results.json` | gpt2 `warm_prefix` prefill tokens |

## Success criteria

- **KV simulation**: prefix-cached row speedup ≫ 1 vs baseline.
- **llmir_paged E2E**: warm-prefix row lower latency vs cold; rising `prefix_hit_tokens`.

## Honesty

- Lengths are tokenizer-accurate when `--model` uses HF tokenizer; word-count fallback is approximate.
- Not a multi-model vLLM throughput table; see E8 (optional empirical) in `PAPER_TOP_TIER_BAR.md`.
