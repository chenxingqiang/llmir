#!/bin/bash
# LLMIR Real Model Benchmark Script
# 在A800 GPU上运行真实模型推理测试

set -e

echo "=============================================="
echo "LLMIR Real Model Inference Benchmark"
echo "=============================================="

# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# 检查GPU
echo ""
echo "=== 检查GPU状态 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 安装依赖
echo ""
echo "=== 安装依赖 ==="
pip install transformers accelerate -q
pip install sentencepiece tiktoken -q

# 创建测试脚本
echo ""
echo "=== 创建测试脚本 ==="

cat > /tmp/benchmark.py << 'BENCHMARK_SCRIPT'
#!/usr/bin/env python3
"""Real Model Benchmark"""
import os
import sys
import time
import torch

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def benchmark_model(model_name, batch_size, seq_len, gen_tokens=128):
    """对单个模型进行基准测试"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Config: batch={batch_size}, seq_len={seq_len}, gen={gen_tokens}")
    print(f"{'='*60}")
    
    # 加载模型
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # 模型信息
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    num_layers = model.config.num_hidden_layers
    print(f"Parameters: {num_params:.2f}B, Layers: {num_layers}")
    
    # 准备输入 - 使用实际文本
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Explain the concept of machine learning in simple terms.

### Response:
"""
    # 扩展到目标长度
    while len(tokenizer.encode(prompt)) < seq_len:
        prompt = prompt + " The quick brown fox jumps over the lazy dog."
    
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt",
                      truncation=True, max_length=seq_len, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_len = inputs['input_ids'].shape[1]
    print(f"Input tokens: {input_len}")
    
    # Warmup
    print("Warmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    
    # 测试生成
    print("Benchmarking...")
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    # 结果
    elapsed = end - start
    output_tokens = outputs.shape[1] - input_len
    total_new_tokens = batch_size * output_tokens
    throughput = total_new_tokens / elapsed
    latency = elapsed / output_tokens * 1000
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    print(f"\n--- Results ---")
    print(f"Generated tokens per sequence: {output_tokens}")
    print(f"Total new tokens: {total_new_tokens}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {throughput:.1f} tokens/s")
    print(f"Latency: {latency:.2f} ms/token")
    print(f"Peak Memory: {peak_mem:.2f} GB")
    
    # 清理
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return {
        'model': model_name.split('/')[-1],
        'params_b': num_params,
        'layers': num_layers,
        'batch': batch_size,
        'input_len': input_len,
        'gen_tokens': output_tokens,
        'time_s': elapsed,
        'throughput': throughput,
        'latency_ms': latency,
        'memory_gb': peak_mem,
    }

def main():
    print("=" * 60)
    print("LLMIR Real Model Inference Benchmark")
    print("=" * 60)
    
    # GPU信息
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_mem:.1f} GB")
    
    results = []
    
    # 测试配置 - 根据80GB A800优化
    test_configs = [
        # (model_name, batch_size, seq_len, gen_tokens)
        # Qwen2.5-7B ~14GB
        ("Qwen/Qwen2.5-7B", 1, 512, 128),
        ("Qwen/Qwen2.5-7B", 4, 512, 128),
        ("Qwen/Qwen2.5-7B", 8, 512, 128),
        ("Qwen/Qwen2.5-7B", 16, 512, 128),
        
        # Phi-3-mini ~7GB
        ("microsoft/Phi-3-mini-4k-instruct", 1, 512, 128),
        ("microsoft/Phi-3-mini-4k-instruct", 8, 512, 128),
        ("microsoft/Phi-3-mini-4k-instruct", 16, 512, 128),
        ("microsoft/Phi-3-mini-4k-instruct", 32, 512, 128),
        
        # GLM-4-9B ~18GB
        ("THUDM/glm-4-9b-chat", 1, 512, 128),
        ("THUDM/glm-4-9b-chat", 4, 512, 128),
        ("THUDM/glm-4-9b-chat", 8, 512, 128),
    ]
    
    for model_name, batch, seq_len, gen_tokens in test_configs:
        try:
            result = benchmark_model(model_name, batch, seq_len, gen_tokens)
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {model_name} batch={batch}, skipping...")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
    
    # 汇总
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Params':>7} {'Layers':>7} {'Batch':>6} {'Throughput':>12} {'Latency':>10} {'Memory':>8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['model']:<25} {r['params_b']:>6.1f}B {r['layers']:>7} {r['batch']:>6} "
              f"{r['throughput']:>10.1f}/s {r['latency_ms']:>8.2f}ms {r['memory_gb']:>6.1f}GB")
    
    # 峰值吞吐量
    if results:
        max_throughput = max(r['throughput'] for r in results)
        best = [r for r in results if r['throughput'] == max_throughput][0]
        print("\n" + "-" * 80)
        print(f"Peak Throughput: {max_throughput:.1f} tokens/s")
        print(f"Best Config: {best['model']} batch={best['batch']}")

if __name__ == "__main__":
    main()
BENCHMARK_SCRIPT

# 运行测试
echo ""
echo "=== 开始基准测试 ==="
python /tmp/benchmark.py

echo ""
echo "=== 测试完成 ==="
