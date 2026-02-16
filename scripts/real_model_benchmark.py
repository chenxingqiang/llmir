#!/usr/bin/env python3
"""
LLMIR Real Model Benchmark
使用真实模型进行推理性能测试
"""

import os
import sys
import time
import torch
import argparse
from typing import List, Dict, Any

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def check_gpu():
    """检查GPU状态"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    return gpu_memory


def benchmark_model_transformers(model_name: str, batch_size: int, seq_len: int, 
                                  num_tokens: int = 128, dtype: str = "float16"):
    """使用transformers进行基准测试"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Batch: {batch_size}, SeqLen: {seq_len}, GenTokens: {num_tokens}")
    print(f"{'='*60}")
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("Loading model...")
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # 获取模型信息
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_params:.2f}B parameters, {num_layers} layers")
    
    # 准备输入
    prompt = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt", 
                       truncation=True, max_length=seq_len, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    actual_seq_len = inputs['input_ids'].shape[1]
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    
    # Benchmark
    print("Running benchmark...")
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 计算指标
    elapsed = end_time - start_time
    total_tokens = batch_size * num_tokens
    throughput = total_tokens / elapsed
    latency_per_token = elapsed / num_tokens * 1000  # ms
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    result = {
        'model': model_name,
        'batch_size': batch_size,
        'input_seq_len': actual_seq_len,
        'gen_tokens': num_tokens,
        'num_layers': num_layers,
        'num_params_b': num_params,
        'elapsed_s': elapsed,
        'throughput_tokens_s': throughput,
        'latency_per_token_ms': latency_per_token,
        'peak_memory_gb': peak_memory,
    }
    
    print(f"\n--- Results ---")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.1f} tokens/s")
    print(f"Latency: {latency_per_token:.2f} ms/token")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    return result


def benchmark_model_vllm(model_name: str, batch_size: int, seq_len: int,
                         num_tokens: int = 128):
    """使用vLLM进行基准测试"""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed, skipping vLLM benchmark")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing with vLLM: {model_name}")
    print(f"Batch: {batch_size}, SeqLen: {seq_len}, GenTokens: {num_tokens}")
    print(f"{'='*60}")
    
    # 创建LLM实例
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.9,
    )
    
    # 准备prompts
    prompt = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10)
    prompts = [prompt] * batch_size
    
    sampling_params = SamplingParams(
        max_tokens=num_tokens,
        temperature=0,
    )
    
    # Warmup
    print("Warming up...")
    _ = llm.generate(prompts[:1], sampling_params)
    
    # Benchmark
    print("Running benchmark...")
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()
    
    # 计算指标
    elapsed = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / elapsed
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    result = {
        'model': model_name,
        'engine': 'vLLM',
        'batch_size': batch_size,
        'gen_tokens': num_tokens,
        'elapsed_s': elapsed,
        'throughput_tokens_s': throughput,
        'peak_memory_gb': peak_memory,
    }
    
    print(f"\n--- vLLM Results ---")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.1f} tokens/s")
    print(f"Peak Memory: {peak_memory:.2f} GB")
    
    return result


def run_comprehensive_benchmark():
    """运行综合基准测试"""
    gpu_memory = check_gpu()
    
    results = []
    
    # 根据GPU内存选择测试模型
    # A800 80GB可以测试的模型
    test_configs = []
    
    if gpu_memory >= 70:
        # 可以测试大模型
        test_configs = [
            # (model_name, batch_size, seq_len, gen_tokens)
            ("Qwen/Qwen2.5-7B", 8, 512, 128),
            ("Qwen/Qwen2.5-7B", 16, 512, 128),
            ("Qwen/Qwen2.5-7B", 32, 256, 128),
            ("microsoft/Phi-3-mini-4k-instruct", 8, 512, 128),
            ("microsoft/Phi-3-mini-4k-instruct", 16, 512, 128),
            ("THUDM/glm-4-9b-chat", 4, 512, 128),
            ("THUDM/glm-4-9b-chat", 8, 512, 128),
        ]
    elif gpu_memory >= 20:
        # 中等内存，测试小模型
        test_configs = [
            ("Qwen/Qwen2.5-1.5B", 8, 512, 128),
            ("Qwen/Qwen2.5-1.5B", 16, 512, 128),
            ("microsoft/Phi-3-mini-4k-instruct", 4, 512, 128),
        ]
    else:
        # 小内存
        test_configs = [
            ("Qwen/Qwen2.5-0.5B", 8, 512, 128),
        ]
    
    print(f"\n{'='*60}")
    print(f"Will test {len(test_configs)} configurations")
    print(f"{'='*60}")
    
    for model_name, batch_size, seq_len, gen_tokens in test_configs:
        try:
            result = benchmark_model_transformers(
                model_name, batch_size, seq_len, gen_tokens
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Batch':>6} {'SeqLen':>7} {'Throughput':>12} {'Memory':>8}")
    print("-" * 80)
    
    for r in results:
        model_short = r['model'].split('/')[-1][:35]
        print(f"{model_short:<40} {r['batch_size']:>6} {r['input_seq_len']:>7} "
              f"{r['throughput_tokens_s']:>10.1f}/s {r['peak_memory_gb']:>6.1f}GB")
    
    return results


def run_single_model(model_name: str, batch_size: int = 8, 
                     seq_len: int = 512, gen_tokens: int = 128):
    """测试单个模型"""
    check_gpu()
    return benchmark_model_transformers(model_name, batch_size, seq_len, gen_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMIR Real Model Benchmark")
    parser.add_argument("--model", type=str, default=None, 
                        help="Specific model to test")
    parser.add_argument("--batch", type=int, default=8, 
                        help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, 
                        help="Input sequence length")
    parser.add_argument("--gen-tokens", type=int, default=128, 
                        help="Number of tokens to generate")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive benchmark")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        run_comprehensive_benchmark()
    elif args.model:
        run_single_model(args.model, args.batch, args.seq_len, args.gen_tokens)
    else:
        # Default: run comprehensive
        run_comprehensive_benchmark()
