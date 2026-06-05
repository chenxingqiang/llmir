#!/usr/bin/env python3
"""
LLMIR Demo - Simulated 0.6B Model Inference

This demo showcases LLMIR's capabilities for optimizing LLM inference:
- PagedKVCache with block-based memory management
- Quantization for memory reduction
- Profiling and performance monitoring
- Model-specific optimizations
- Continuous batching engine

Note: This is a simulation demo. For actual model inference,
integrate with PyTorch/Transformers.
"""

import numpy as np
import time
import sys

# Add src to path for development
sys.path.insert(0, '/workspace/src')

import llmir
from llmir import (
    PagedKVCache,
    QuantizedKVCache,
    SpeculativeKVCache,
    PrefixCache,
    KVCacheConfig,
    QuantizationConfig,
    QuantizationType,
    SpeculativeConfig,
    LLMEngine,
    ContinuousBatchingEngine,
    SamplingParams,
    SchedulerConfig,
    SchedulingPolicy,
    Profiler,
    MemoryProfiler,
    LatencyProfiler,
    ThroughputMonitor,
    ModelOptimizer,
    ModelRegistry,
    ModelMemoryEstimator,
)
from llmir.models import ModelConfig, ModelArchitecture, AttentionType


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_model_config():
    """Demo: Configure a 0.6B model."""
    print_header("1. Model Configuration (0.6B Model)")
    
    # Define a Qwen-0.5B like configuration
    config = ModelConfig(
        architecture=ModelArchitecture.QWEN,
        attention_type=AttentionType.GROUPED_QUERY,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=2,  # GQA with 8:1 ratio
        head_dim=64,
        intermediate_size=2816,
        vocab_size=151936,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    )
    
    print(f"Model Architecture: {config.architecture.name}")
    print(f"Attention Type: {config.attention_type.name}")
    print(f"Layers: {config.num_layers}")
    print(f"Hidden Size: {config.hidden_size}")
    print(f"Attention Heads: {config.num_attention_heads}")
    print(f"KV Heads: {config.num_key_value_heads} (GQA ratio: {config.get_num_queries_per_kv()}:1)")
    print(f"Head Dim: {config.get_head_dim()}")
    print(f"Max Position: {config.max_position_embeddings}")
    
    # Memory estimation
    estimator = ModelMemoryEstimator(config)
    
    print(f"\n--- Memory Estimation ---")
    weight_mem = estimator.estimate_weight_memory("float16")
    print(f"Model Weights (FP16): {weight_mem / 1e9:.2f} GB")
    
    # KV cache for different batch sizes
    for batch_size in [1, 8, 32]:
        seq_len = 2048
        kv_mem = estimator.estimate_kv_cache_memory(batch_size, seq_len, "float16")
        print(f"KV Cache (batch={batch_size}, seq={seq_len}): {kv_mem / 1e6:.2f} MB")
    
    return config


def demo_kv_cache(model_config: ModelConfig):
    """Demo: PagedKVCache operations."""
    print_header("2. PagedKVCache Demo")
    
    # Create optimized KV cache config
    kv_config = KVCacheConfig(
        num_layers=model_config.num_layers,
        num_heads=model_config.num_key_value_heads,
        head_dim=model_config.get_head_dim(),
        block_size=16,
        max_seq_len=4096,
        dtype="float16",
        enable_gpu=False,  # CPU mode for demo
    )
    
    print(f"KV Cache Configuration:")
    print(f"  Layers: {kv_config.num_layers}")
    print(f"  KV Heads: {kv_config.num_heads}")
    print(f"  Head Dim: {kv_config.head_dim}")
    print(f"  Block Size: {kv_config.block_size}")
    
    # Create cache
    cache = PagedKVCache(kv_config)
    
    # Simulate prefill
    batch_size = 4
    prefill_len = 128
    
    print(f"\n--- Simulating Prefill ---")
    print(f"Batch Size: {batch_size}")
    print(f"Prefill Length: {prefill_len}")
    
    # Create dummy KV tensors
    keys = np.random.randn(batch_size, prefill_len, 
                          kv_config.num_heads, kv_config.head_dim).astype(np.float16)
    values = np.random.randn(batch_size, prefill_len,
                            kv_config.num_heads, kv_config.head_dim).astype(np.float16)
    seq_ids = np.arange(batch_size, dtype=np.int32)
    
    start = time.perf_counter()
    block_indices = cache.append(keys, values, seq_ids)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"Prefill completed in {elapsed:.2f} ms")
    print(f"Block indices shape: {block_indices.shape}")
    print(f"Active sequences: {cache.get_num_sequences()}")
    
    # Simulate decode steps
    print(f"\n--- Simulating Decode (100 tokens) ---")
    decode_times = []
    
    for step in range(100):
        keys = np.random.randn(batch_size, 1, 
                              kv_config.num_heads, kv_config.head_dim).astype(np.float16)
        values = np.random.randn(batch_size, 1,
                                kv_config.num_heads, kv_config.head_dim).astype(np.float16)
        
        start = time.perf_counter()
        cache.append(keys, values, seq_ids)
        decode_times.append((time.perf_counter() - start) * 1000)
    
    avg_decode = np.mean(decode_times)
    print(f"Average decode step: {avg_decode:.3f} ms")
    print(f"Tokens/second: {1000 / avg_decode:.1f}")
    
    return cache


def demo_quantization(model_config: ModelConfig):
    """Demo: Quantized KV Cache."""
    print_header("3. Quantized KV Cache Demo")
    
    kv_config = KVCacheConfig(
        num_layers=model_config.num_layers,
        num_heads=model_config.num_key_value_heads,
        head_dim=model_config.get_head_dim(),
        block_size=16,
        max_seq_len=4096,
    )
    
    # Compare FP16 vs INT8 vs INT4
    print("Memory comparison (per sequence, 2048 tokens):")
    print("-" * 45)
    
    # FP16 baseline
    fp16_memory = (2 * model_config.num_layers * model_config.num_key_value_heads * 
                   model_config.get_head_dim() * 2048 * 2)  # 2 bytes per FP16
    print(f"FP16:  {fp16_memory / 1e6:>8.2f} MB (1.0x)")
    
    # INT8
    quant_config_int8 = QuantizationConfig(quant_type=QuantizationType.INT8)
    cache_int8 = QuantizedKVCache(kv_config, quant_config_int8)
    ratio_int8 = cache_int8.get_compression_ratio()
    print(f"INT8:  {fp16_memory / ratio_int8 / 1e6:>8.2f} MB ({ratio_int8:.1f}x compression)")
    
    # INT4
    quant_config_int4 = QuantizationConfig(quant_type=QuantizationType.INT4)
    cache_int4 = QuantizedKVCache(kv_config, quant_config_int4)
    ratio_int4 = cache_int4.get_compression_ratio()
    print(f"INT4:  {fp16_memory / ratio_int4 / 1e6:>8.2f} MB ({ratio_int4:.1f}x compression)")
    
    # Accuracy estimation
    print("\nEstimated accuracy impact:")
    print(f"INT8: ~{cache_int8.get_accuracy_loss() * 100:.2f}% loss")
    print(f"INT4: ~{cache_int4.get_accuracy_loss() * 100:.2f}% loss")


def demo_speculative_decoding(model_config: ModelConfig):
    """Demo: Speculative Decoding with KV Cache."""
    print_header("4. Speculative Decoding Demo")
    
    kv_config = KVCacheConfig(
        num_layers=model_config.num_layers,
        num_heads=model_config.num_key_value_heads,
        head_dim=model_config.get_head_dim(),
        block_size=16,
        max_seq_len=4096,
    )
    
    spec_config = SpeculativeConfig(
        max_draft_tokens=8,
        max_branches=4,
        enable_tree_attention=True,
        acceptance_threshold=0.9,
    )
    
    print(f"Speculative Config:")
    print(f"  Max Draft Tokens: {spec_config.max_draft_tokens}")
    print(f"  Max Branches: {spec_config.max_branches}")
    print(f"  Tree Attention: {spec_config.enable_tree_attention}")
    
    cache = SpeculativeKVCache(kv_config, spec_config)
    
    # Simulate speculative decoding
    print("\n--- Simulating Speculative Decoding ---")
    
    seq_id = 0
    total_accepted = 0
    total_drafted = 0
    
    for iteration in range(10):
        # Create branch for speculation
        branch_id = cache.create_branch(seq_id)
        
        # Simulate drafting 8 tokens
        draft_tokens = 8
        total_drafted += draft_tokens
        
        # Simulate acceptance (random for demo)
        accepted = np.random.randint(3, 8)  # Accept 3-8 tokens
        total_accepted += accepted
        
        # Commit or rollback
        cache.commit(seq_id, branch_id, accepted)
        
        print(f"  Iteration {iteration + 1}: drafted {draft_tokens}, accepted {accepted}")
    
    acceptance_rate = total_accepted / total_drafted
    speedup = 1 + (acceptance_rate * (spec_config.max_draft_tokens - 1))
    
    print(f"\nResults:")
    print(f"  Total Drafted: {total_drafted}")
    print(f"  Total Accepted: {total_accepted}")
    print(f"  Acceptance Rate: {acceptance_rate:.1%}")
    print(f"  Estimated Speedup: {speedup:.2f}x")


def demo_prefix_caching():
    """Demo: Prefix Caching for system prompts."""
    print_header("5. Prefix Caching Demo")
    
    cache = PrefixCache()
    
    # Simulate system prompt tokens
    system_prompt = list(range(1000, 1512))  # 512 tokens
    block_indices = [[i] for i in range(32)]  # 32 blocks
    
    print(f"Caching system prompt ({len(system_prompt)} tokens)...")
    cache.cache_prefix(system_prompt, block_indices)
    
    # Simulate multiple requests with same prefix
    print("\n--- Processing Requests ---")
    
    for i in range(5):
        # Request with system prompt + user query
        user_query = list(range(2000 + i * 100, 2000 + i * 100 + 50))
        full_prompt = system_prompt + user_query
        
        match_len, cached_blocks = cache.lookup(full_prompt)
        
        if match_len > 0:
            print(f"Request {i + 1}: Matched {match_len} tokens, "
                  f"skipped {match_len} tokens of KV computation")
        else:
            print(f"Request {i + 1}: No prefix match, full computation needed")
    
    print(f"\nCache Statistics:")
    stats = cache.get_stats()
    print(f"  Cached Prefixes: {stats['num_prefixes']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Hit Ratio: {stats['hit_ratio']:.1%}")


def demo_continuous_batching():
    """Demo: Continuous Batching Engine."""
    print_header("6. Continuous Batching Demo")
    
    kv_config = KVCacheConfig(
        num_layers=24,
        num_heads=2,
        head_dim=64,
        block_size=16,
        max_seq_len=4096,
    )
    
    cache = PagedKVCache(kv_config)
    
    scheduler_config = SchedulerConfig(
        policy=SchedulingPolicy.ADAPTIVE,
        max_batch_size=32,
        max_num_seqs=64,
        enable_preemption=True,
    )
    
    engine = ContinuousBatchingEngine(cache, scheduler_config)
    engine.start()
    
    print(f"Engine Configuration:")
    print(f"  Policy: {scheduler_config.policy.name}")
    print(f"  Max Batch Size: {scheduler_config.max_batch_size}")
    print(f"  Preemption: {'Enabled' if scheduler_config.enable_preemption else 'Disabled'}")
    
    # Submit requests
    print("\n--- Submitting Requests ---")
    
    prompts = [
        [1, 2, 3, 4, 5],           # Short prompt
        list(range(100)),          # Medium prompt
        list(range(500)),          # Long prompt
        [10, 20, 30],              # Short prompt
        list(range(200)),          # Medium prompt
    ]
    
    for i, prompt in enumerate(prompts):
        params = SamplingParams(max_tokens=50, temperature=0.7)
        req_id = engine.submit(prompt, params)
        print(f"  Submitted request {req_id} ({len(prompt)} tokens)")
    
    # Process requests
    print("\n--- Processing Requests ---")
    
    iterations = 0
    while engine.has_pending_requests() and iterations < 100:
        outputs = engine.iterate()
        iterations += 1
        
        for output in outputs:
            if output.finished:
                print(f"  Request {output.request_id} completed "
                      f"({len(output.outputs[0].token_ids)} tokens)")
    
    # Statistics
    stats = engine.get_stats()
    print(f"\nEngine Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Completed: {stats['completed_requests']}")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/sec")
    
    engine.stop()


def demo_profiling():
    """Demo: Performance Profiling."""
    print_header("7. Performance Profiling Demo")
    
    profiler = Profiler()
    latency_profiler = LatencyProfiler()
    throughput_monitor = ThroughputMonitor()
    
    profiler.start()
    throughput_monitor.start()
    
    # Simulate workload
    print("Running profiled workload...")
    
    for i in range(50):
        # Simulate prefill
        with profiler.trace("prefill"):
            with latency_profiler.measure("prefill"):
                time.sleep(0.002)  # 2ms
        
        # Simulate decode
        for j in range(10):
            with profiler.trace("decode"):
                with latency_profiler.measure("decode"):
                    time.sleep(0.0005)  # 0.5ms
            throughput_monitor.record_tokens(1)
        
        throughput_monitor.record_request(10)
    
    profiler.stop()
    throughput_monitor.stop()
    
    # Print results
    print("\n--- Profiling Results ---")
    report = profiler.get_report()
    report.print_summary()
    
    print("\n--- Latency Statistics ---")
    latency_stats = latency_profiler.get_stats()
    for name, stats in latency_stats.items():
        print(f"\n{name}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.3f} ms")
        print(f"  P50: {stats['p50']:.3f} ms")
        print(f"  P99: {stats['p99']:.3f} ms")
    
    print("\n--- Throughput Statistics ---")
    tp_stats = throughput_monitor.get_stats()
    print(f"Total Tokens: {tp_stats['total_tokens']}")
    print(f"Total Requests: {tp_stats['total_requests']}")
    print(f"Tokens/sec: {tp_stats['tokens_per_second']:.1f}")
    print(f"Requests/sec: {tp_stats['requests_per_second']:.1f}")


def demo_model_registry():
    """Demo: Model Registry."""
    print_header("8. Model Registry Demo")
    
    registry = ModelRegistry()
    
    print("Available pre-configured models:")
    print("-" * 40)
    
    for model_name in sorted(registry.list_models()):
        config = registry.get(model_name)
        if config:
            # Estimate size
            estimator = ModelMemoryEstimator(config)
            size_gb = estimator.estimate_weight_memory("float16") / 1e9
            print(f"  {model_name:<20} ~{size_gb:.1f}GB")
    
    # Get optimizer for specific model
    print("\n--- Llama 3.1 8B Optimization ---")
    optimizer = registry.get_optimizer("llama3.1-8b")
    if optimizer:
        kv_config = optimizer.get_optimized_kv_cache_config()
        print(f"Optimized KV Cache:")
        print(f"  Block Size: {kv_config.block_size}")
        print(f"  KV Heads: {kv_config.num_heads}")
        
        max_batch = optimizer.get_recommended_batch_size(gpu_memory_gb=24)
        print(f"  Recommended Batch Size (24GB GPU): {max_batch}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  LLMIR Demo - 0.6B Model Simulation")
    print(f"  Version: {llmir.__version__}")
    print("=" * 60)
    
    # Run demos
    model_config = demo_model_config()
    demo_kv_cache(model_config)
    demo_quantization(model_config)
    demo_speculative_decoding(model_config)
    demo_prefix_caching()
    demo_continuous_batching()
    demo_profiling()
    demo_model_registry()
    
    print_header("Demo Complete!")
    print("\nLLMIR provides:")
    print("  ✓ PagedKVCache for efficient memory management")
    print("  ✓ INT8/INT4 quantization for 4-8x compression")
    print("  ✓ Speculative decoding for faster generation")
    print("  ✓ Prefix caching for prompt reuse")
    print("  ✓ Continuous batching for production serving")
    print("  ✓ Comprehensive profiling tools")
    print("  ✓ Model-specific optimizations")
    print("\nFor more info: https://chenxingqiang.github.io/llmir-www/")


if __name__ == "__main__":
    main()
