"""
LLMIR Command Line Interface

Command-line tools for profiling, benchmarking, and listing models.
"""

import argparse
import sys


def list_models_main():
    """Main entry point for llmir-list-models command."""
    parser = argparse.ArgumentParser(
        description="List LLMIR registry models and supported HuggingFace architectures",
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="Show only built-in registry model names (one per line)",
    )
    args = parser.parse_args()

    from llmir.models import ModelRegistry
    registry = ModelRegistry()
    models = sorted(registry.list_models())

    if args.registry_only:
        for m in models:
            print(m)
        return 0

    print("LLMIR Supported Models")
    print("=" * 50)
    print("Built-in registry:")
    for m in models:
        print(f"  {m}")
    print()
    print("HuggingFace: any decoder-only model (pip install llmir[full])")
    print("  Examples: meta-llama/Llama-3.1-8B, Qwen/Qwen2-0.5B")
    return 0


def profile_main():
    """Main entry point for llmir-profile command."""
    parser = argparse.ArgumentParser(
        description="LLMIR Performance Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmir-profile --model llama-7b --batch-size 8 --seq-len 512
  llmir-profile --input trace.json --output report.html
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name or path to profile"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for profiling"
    )
    parser.add_argument(
        "--seq-len", "-s",
        type=int,
        default=512,
        help="Sequence length for profiling"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input trace file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="profile_report.json",
        help="Output report file"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "chrome", "text"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--export-config",
        type=str,
        metavar="PATH",
        help="Export model and KV cache config as JSON to file",
    )
    
    args = parser.parse_args()
    
    print("LLMIR Profiler")
    print("=" * 40)
    
    if args.model:
        from llmir.models import ModelRegistry
        registry = ModelRegistry()
        optimizer = None

        if registry.has_model(args.model):
            optimizer = registry.get_optimizer(args.model)
        else:
            try:
                from llmir import from_pretrained
                if from_pretrained:
                    optimizer = from_pretrained(args.model)
            except (ImportError, Exception):
                pass

        if optimizer:
            print(f"Model: {args.model}")
            print(f"Batch size: {args.batch_size}")
            print(f"Sequence length: {args.seq_len}")
            memory = optimizer.estimate_memory(args.batch_size, args.seq_len)
            print(f"Estimated memory: {memory / 1e9:.2f} GB")
            if args.export_config:
                import json
                kv = optimizer.get_optimized_kv_cache_config()
                out = {
                    "model": args.model,
                    "model_config": optimizer.config.to_dict(),
                    "kv_cache_config": kv.to_dict(),
                }
                with open(args.export_config, "w") as f:
                    json.dump(out, f, indent=2)
                print(f"Config exported to {args.export_config}")
        else:
            print(f"Unknown model: {args.model}")
            print(f"Registry models: {', '.join(registry.list_models())}")
            print("For HuggingFace models, install: pip install llmir[full]")
    else:
        print("No model specified. Use --model to specify a model.")
        print(f"Available models: {', '.join(ModelRegistry().list_models())}")
    
    return 0


def benchmark_main():
    """Main entry point for llmir-benchmark command."""
    import time
    import json
    import numpy as np

    from llmir.models import ModelRegistry
    from llmir import PagedKVCache

    parser = argparse.ArgumentParser(
        description="LLMIR Benchmark Tool - KV cache and config benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llmir-benchmark --model llama3-8b --batch-sizes 1,4,8
  llmir-benchmark --model llama3-8b --output results.json
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name from registry (e.g. llama3-8b) or HuggingFace ID (requires llmir[full])"
    )
    parser.add_argument(
        "--batch-sizes", "-b",
        type=str,
        default="1,2,4,8",
        help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--seq-lens", "-s",
        type=str,
        default="128,512,1024",
        help="Comma-separated sequence lengths"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=3,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=50,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.json",
        help="Output results file (JSON)"
    )
    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]

    print("LLMIR Benchmark (KV Cache + Config)")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    optimizer = None
    registry = ModelRegistry()
    if registry.has_model(args.model):
        optimizer = registry.get_optimizer(args.model)
    else:
        try:
            from llmir import from_pretrained
            if from_pretrained:
                optimizer = from_pretrained(args.model)
        except ImportError:
            pass
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
    if not optimizer:
        print(f"Unknown model: {args.model}")
        print(f"Registry models: {', '.join(registry.list_models())}")
        return 1

    kv_config = optimizer.get_optimized_kv_cache_config()
    mem_est = optimizer.estimate_memory(1, 512)
    print(f"KV config: layers={kv_config.num_layers}, heads={kv_config.num_heads}, "
          f"head_dim={kv_config.head_dim}, block_size={kv_config.block_size}")
    print(f"Memory estimate (batch=1, seq=512): {mem_est / 1e9:.3f} GB")
    print()

    results = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            if batch_size * seq_len * kv_config.num_heads * kv_config.head_dim > 50_000_000:
                print(f"Skipping batch={batch_size} seq={seq_len} (large)")
                continue
            cache = PagedKVCache(kv_config)
            keys = np.random.randn(batch_size, seq_len, kv_config.num_heads, kv_config.head_dim).astype(np.float16)
            values = np.random.randn(batch_size, seq_len, kv_config.num_heads, kv_config.head_dim).astype(np.float16)
            seq_ids = np.arange(batch_size, dtype=np.int32)
            for _ in range(args.warmup):
                cache.append(keys, values, seq_ids)
                cache.reset()
            start = time.perf_counter()
            for _ in range(args.iterations):
                bi = cache.append(keys, values, seq_ids)
                k, v = cache.lookup(bi, np.full(batch_size, seq_len))
                cache.reset()
            elapsed = time.perf_counter() - start
            tokens = batch_size * seq_len * args.iterations * 2
            throughput = tokens / elapsed
            r = {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "time_s": round(elapsed, 3),
                "throughput_tokens_per_s": round(throughput, 1),
                "iterations": args.iterations,
            }
            results.append(r)
            print(f"  batch={batch_size:2d} seq={seq_len:4d}: {throughput:,.0f} tok/s")

    out = {"model": args.model, "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(profile_main())
