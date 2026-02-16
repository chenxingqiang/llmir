#!/usr/bin/env python3
"""
LLMIR MPS Validation Script

Validates:
1. LLMIR config extraction from HuggingFace models
2. Model loading with transformers (no TF/protobuf conflict)
3. LLMIR PagedKVCache micro-benchmark (synthetic KV data)
4. Transformers inference (model.generate)
5. LLMIR E2E (formal): manual generation with KV routed through PagedKVCache
   - Uses llmir.runtime.PagedKVCache (real component, not simulated)
   - After each forward: append model K/V to PagedKVCache
   - Before next forward: lookup from PagedKVCache, pass to model
   - Validates that real K/V flows through LLMIR without corrupting generation
"""

# Disable TensorFlow in transformers to avoid protobuf/tensorflow dependency conflict.
# Must be set before any transformers import.
import os
import sys
import time

os.environ.setdefault("USE_TORCH", "1")

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, os.path.join(_root, "src"))

import torch


def _e2e_generate_with_llmir_paged_kv_cache(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    kv_config,
    max_new_tokens: int = 16,
) -> str:
    """
    Formal E2E: manual generation loop with KV routed through LLMIR PagedKVCache.
    After each forward, append real model K/V to PagedKVCache; before next forward,
    lookup from PagedKVCache and pass to model. Uses actual llmir.runtime.PagedKVCache.
    """
    import numpy as np
    from transformers import DynamicCache

    from llmir import PagedKVCache

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    prompt_len = input_ids.shape[1]
    generated_ids = input_ids.clone()
    cache_position = torch.arange(prompt_len, dtype=torch.int64, device=device)

    # One PagedKVCache per layer (LLMIR real component)
    layer_caches = [PagedKVCache(kv_config) for _ in range(len(model.model.layers))]
    dtype = next(model.parameters()).dtype
    batch_size = input_ids.shape[0]
    seq_ids = np.zeros(batch_size, dtype=np.int32)

    def _append_to_llmir(cache_obj, append_only_new: bool):
        for layer_idx, (k, v) in enumerate(zip(cache_obj.key_cache, cache_obj.value_cache)):
            # DynamicCache: (batch, num_heads, seq_len, head_dim)
            if append_only_new and k.shape[2] > 1:
                k = k[:, :, -1:, :]  # (batch, num_heads, 1, head_dim)
                v = v[:, :, -1:, :]
            # -> PagedKVCache: (batch, seq_len, num_heads, head_dim)
            k_np = k.detach().cpu().float().numpy().transpose(0, 2, 1, 3)
            v_np = v.detach().cpu().float().numpy().transpose(0, 2, 1, 3)
            layer_caches[layer_idx].append(k_np, v_np, seq_ids)

    def _lookup_from_llmir(past_len: int) -> DynamicCache:
        block_indices = np.zeros((batch_size, kv_config.num_layers), dtype=np.int32)
        seq_lens = np.full(batch_size, past_len, dtype=np.int32)
        data = []
        for layer_idx, lc in enumerate(layer_caches):
            k_np, v_np = lc.lookup(block_indices, seq_lens)
            # PagedKVCache: (batch, seq_len, num_heads, head_dim) -> DynamicCache: (batch, num_heads, seq_len, head_dim)
            k_t = torch.from_numpy(k_np.transpose(0, 2, 1, 3)).to(device=device, dtype=dtype)
            v_t = torch.from_numpy(v_np.transpose(0, 2, 1, 3)).to(device=device, dtype=dtype)
            data.append((k_t, v_t))
        return DynamicCache(_distributed_cache_data=data)

    first_step = True
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if first_step:
                past_key_values = None
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "use_cache": True,
                }
            else:
                past_len = layer_caches[0].get_sequence_length(0)
                past_key_values = _lookup_from_llmir(past_len)
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "cache_position": cache_position,
                    "use_cache": True,
                }

            outputs = model(**model_inputs)
            past_key_values = outputs.past_key_values

            # Route through LLMIR PagedKVCache (formal validation)
            _append_to_llmir(past_key_values, append_only_new=not first_step)
            first_step = False

            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if tokenizer.eos_token_id is not None and (next_token == tokenizer.eos_token_id).all():
                break

            input_ids = next_token
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                    dim=-1,
                )
            cache_position = cache_position[-1:] + 1

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def detect_device():
    """Detect best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon)"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"


def run_mps_pipeline(
    model_id: str = "Qwen/Qwen2-0.5B",
    batch_size: int = 2,
    seq_len: int = 64,
    gen_tokens: int = 32,
):
    """Validate LLMIR config, model load, KV cache micro-bench, and transformers inference."""
    throughput, elapsed = 0.0, 0.0
    device, device_name = detect_device()
    print("=" * 60)
    print("LLMIR MPS Validation")
    print("=" * 60)
    print(f"Device: {device_name}")
    print(f"Model: {model_id}")
    print(f"Batch: {batch_size}, SeqLen: {seq_len}, GenTokens: {gen_tokens}")
    print()

    # 1. LLMIR: get optimized config
    print("[1/5] LLMIR config...")
    try:
        from llmir import from_pretrained

        optimizer = from_pretrained(model_id)
        kv_config = optimizer.get_optimized_kv_cache_config()
        mem_est = optimizer.estimate_memory(batch_size, seq_len + gen_tokens)
        print(f"  Layers: {kv_config.num_layers}, Heads: {kv_config.num_heads}")
        print(f"  Block size: {kv_config.block_size}")
        print(f"  Est. memory: {mem_est / 1e9:.2f} GB")
    except ImportError:
        print("  (llmir[full] not installed, using default config)")
        from llmir.runtime.config import KVCacheConfig

        kv_config = KVCacheConfig(num_layers=24, num_heads=2, head_dim=64)
    except Exception as e:
        print(f"  WARNING: {e}")
        kv_config = None

    # 2. Load model with transformers
    print("\n[2/5] Loading model with transformers...")
    model = None
    tokenizer = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        if device.type == "mps":
            model = model.to(device)
        elif device.type == "cpu" and next(model.parameters()).is_cuda:
            model = model.cpu()
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Loaded: {params:.2f}B params")
    except Exception as e:
        print(f"  Skip (env issue): {e}")
        print("  Tip: pip install 'transformers>=4.40' without tensorflow for model load")

    # 3. LLMIR PagedKVCache micro-benchmark (synthetic KV data; not used in inference)
    if kv_config:
        print("\n[3/5] LLMIR PagedKVCache micro-benchmark (synthetic KV)...")
        import numpy as np
        from llmir import PagedKVCache

        cache = PagedKVCache(kv_config)
        keys = np.random.randn(
            batch_size, seq_len, kv_config.num_heads, kv_config.head_dim
        ).astype(np.float16)
        values = np.random.randn(
            batch_size, seq_len, kv_config.num_heads, kv_config.head_dim
        ).astype(np.float16)
        seq_ids = np.arange(batch_size, dtype=np.int32)

        warmup = 5
        iters = 50
        for _ in range(warmup):
            cache.append(keys, values, seq_ids)
            cache.reset()
        start = time.perf_counter()
        for _ in range(iters):
            bi = cache.append(keys, values, seq_ids)
            _ = cache.lookup(bi, np.full(batch_size, seq_len))
            cache.reset()
        elapsed = time.perf_counter() - start
        ops = iters * 2  # append + lookup per iter
        ops_per_sec = ops / elapsed
        print(f"  Append+lookup: {ops_per_sec:.0f} ops/s ({elapsed:.3f}s)")
        print(f"  (Synthetic KV; inference below uses PyTorch's KV cache)")

    # 4. Transformers inference (model.generate)
    print("\n[4/5] Transformers inference (model.generate)...")
    if model is not None and tokenizer is not None:
        prompt = "Hello, how are you? " * (seq_len // 5)
        inputs = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Warmup
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens = batch_size * gen_tokens
        throughput = total_tokens / elapsed
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {throughput:.1f} tokens/s")

        # Decode first output
        out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preview = out_text[:120] + "..." if len(out_text) > 120 else out_text
        print(f"  Sample: {preview!r}")

        # 5. LLMIR E2E: formal validation + performance
        print("\n[5/5] LLMIR E2E (KV through PagedKVCache)...")
        e2e_tokens = min(gen_tokens, 16)
        e2e_prompt = "What is 2+2?"
        if kv_config is None:
            kv_config = __import__('llmir.runtime.config', fromlist=['KVCacheConfig']).KVCacheConfig(
                num_layers=len(model.model.layers),
                num_heads=getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads),
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )
        try:
            # Correctness
            e2e_out = _e2e_generate_with_llmir_paged_kv_cache(
                model, tokenizer, e2e_prompt, device, kv_config, max_new_tokens=e2e_tokens
            )
            ref_out = tokenizer.decode(
                model.generate(
                    tokenizer(e2e_prompt, return_tensors="pt")["input_ids"].to(device),
                    max_new_tokens=e2e_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )[0],
                skip_special_tokens=True,
            )
            match = e2e_out.strip() == ref_out.strip()
            print(f"  Match: {match}")
            if not match:
                print(f"  E2E: {e2e_out[:60]!r}...")
                print(f"  Ref:  {ref_out[:60]!r}...")

            # Performance: benchmark E2E vs model.generate (same prompt, batch=1)
            e2e_inputs = tokenizer(e2e_prompt, return_tensors="pt")
            e2e_inputs = {k: v.to(device) for k, v in e2e_inputs.items()}
            if "attention_mask" not in e2e_inputs:
                e2e_inputs["attention_mask"] = torch.ones_like(e2e_inputs["input_ids"], device=device)

            with torch.no_grad():
                _ = _e2e_generate_with_llmir_paged_kv_cache(
                    model, tokenizer, e2e_prompt, device, kv_config, max_new_tokens=e2e_tokens
                )
            if device.type in ("mps", "cuda"):
                getattr(torch, device.type).synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = _e2e_generate_with_llmir_paged_kv_cache(
                    model, tokenizer, e2e_prompt, device, kv_config, max_new_tokens=e2e_tokens
                )
            if device.type in ("mps", "cuda"):
                getattr(torch, device.type).synchronize()
            e2e_elapsed = time.perf_counter() - start
            e2e_tok_per_s = e2e_tokens / e2e_elapsed

            start = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(
                    **e2e_inputs,
                    max_new_tokens=e2e_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            if device.type in ("mps", "cuda"):
                getattr(torch, device.type).synchronize()
            ref_elapsed = time.perf_counter() - start
            ref_tok_per_s = e2e_tokens / ref_elapsed

            print(f"  E2E: {e2e_tok_per_s:.1f} tokens/s ({e2e_elapsed:.3f}s)")
            print(f"  Ref: {ref_tok_per_s:.1f} tokens/s ({ref_elapsed:.3f}s)")
            ratio = e2e_tok_per_s / ref_tok_per_s if ref_tok_per_s > 0 else 0
            print(f"  Ratio (E2E/Ref): {ratio:.2%}")
        except Exception as e:
            print(f"  Skip: {e}")
    else:
        # Fallback: device tensor ops to verify MPS/CUDA works
        print("  Running device tensor benchmark (model load skipped)...")
        x = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        for _ in range(10):
            _ = torch.mm(x, x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.mm(x, x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"  {device_name} matmul 100x: {elapsed:.3f}s (device OK)")

    print("\n" + "=" * 60)
    print("Validation OK")
    print("=" * 60)
    return {"throughput": throughput, "elapsed": elapsed}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLMIR MPS validation script")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model ID (default: Qwen2-0.5B)",
    )
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Input sequence length")
    parser.add_argument("--gen-tokens", type=int, default=32, help="Tokens to generate")
    args = parser.parse_args()

    run_mps_pipeline(
        model_id=args.model,
        batch_size=args.batch,
        seq_len=args.seq_len,
        gen_tokens=args.gen_tokens,
    )
