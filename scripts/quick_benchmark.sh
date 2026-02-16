#!/bin/bash
# Quick Single Model Benchmark
# 快速单模型测试脚本

echo "=============================================="
echo "Quick Model Benchmark"
echo "=============================================="

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 安装依赖
pip install transformers accelerate sentencepiece -q 2>/dev/null

python3 << 'EOF'
import os
import time
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForCausalLM, AutoTokenizer

def quick_benchmark(model_name="Qwen/Qwen2.5-1.5B"):
    print(f"\n=== Testing {model_name} ===")
    
    # GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    params = sum(p.numel() for p in model.parameters()) / 1e9
    layers = model.config.num_hidden_layers
    print(f"Loaded: {params:.2f}B params, {layers} layers")
    
    # Test different batch sizes
    results = []
    for batch_size in [1, 4, 8, 16, 32]:
        try:
            prompt = "Explain machine learning: " * 50
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt",
                             truncation=True, max_length=512, padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Warmup
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            
            # Benchmark
            gen_tokens = 128
            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=gen_tokens, 
                                   pad_token_id=tokenizer.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            throughput = batch_size * gen_tokens / elapsed
            results.append((batch_size, throughput, elapsed))
            print(f"  Batch {batch_size:>2}: {throughput:>8.1f} tokens/s  ({elapsed:.2f}s)")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {batch_size:>2}: OOM")
            torch.cuda.empty_cache()
            break
    
    if results:
        best_batch, best_throughput, _ = max(results, key=lambda x: x[1])
        print(f"\n  Peak: {best_throughput:.1f} tokens/s (batch={best_batch})")
    
    return results

if __name__ == "__main__":
    # 测试小模型验证环境
    quick_benchmark("Qwen/Qwen2.5-1.5B")
    
    torch.cuda.empty_cache()
    
    # 测试中等模型
    quick_benchmark("Qwen/Qwen2.5-7B")
EOF
