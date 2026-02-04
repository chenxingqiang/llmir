#!/bin/bash
# PyTorch vs vLLM Direct Comparison
# 最简洁的对比测试

set -e

echo "=============================================="
echo "PyTorch vs vLLM Performance Comparison"
echo "=============================================="

export HF_ENDPOINT=https://hf-mirror.com

# 安装
echo "Installing dependencies..."
pip install torch transformers accelerate -q
pip install vllm -q

cat > /tmp/vllm_compare.py << 'EOF'
import os
import time
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def pytorch_benchmark(model_name, batch_sizes, seq_len=512, gen_len=128):
    """PyTorch transformers baseline"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print(f"[PyTorch] {model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e9
    layers = model.config.num_hidden_layers
    print(f"Model: {params:.2f}B params, {layers} layers")
    
    results = []
    prompt = "Explain the concept of artificial intelligence and its applications: " * 20
    
    for batch_size in batch_sizes:
        try:
            inputs = tokenizer([prompt] * batch_size, return_tensors="pt",
                             truncation=True, max_length=seq_len, padding=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            input_len = inputs['input_ids'].shape[1]
            
            # Warmup
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            start = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=gen_len,
                                   pad_token_id=tokenizer.pad_token_id, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            gen_tokens = out.shape[1] - input_len
            throughput = batch_size * gen_tokens / elapsed
            memory = torch.cuda.max_memory_allocated() / 1e9
            
            results.append({
                'batch': batch_size,
                'throughput': throughput,
                'memory': memory,
                'time': elapsed
            })
            print(f"  Batch {batch_size:>2}: {throughput:>8.1f} tok/s | {memory:.1f}GB | {elapsed:.2f}s")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {batch_size:>2}: OOM")
            torch.cuda.empty_cache()
            break
    
    del model, tokenizer
    torch.cuda.empty_cache()
    return results


def vllm_benchmark(model_name, batch_sizes, seq_len=512, gen_len=128):
    """vLLM benchmark"""
    from vllm import LLM, SamplingParams
    
    print(f"\n{'='*60}")
    print(f"[vLLM] {model_name}")
    print(f"{'='*60}")
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
    )
    
    results = []
    prompt = "Explain the concept of artificial intelligence and its applications: " * 20
    # Truncate prompt to approximate seq_len tokens
    prompt = prompt[:seq_len * 4]  # rough char to token ratio
    
    for batch_size in batch_sizes:
        try:
            prompts = [prompt] * batch_size
            sampling_params = SamplingParams(max_tokens=gen_len, temperature=0)
            
            # Warmup
            _ = llm.generate(prompts[:1], SamplingParams(max_tokens=5, temperature=0))
            torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - start
            
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed
            memory = torch.cuda.max_memory_allocated() / 1e9
            
            results.append({
                'batch': batch_size,
                'throughput': throughput,
                'memory': memory,
                'time': elapsed
            })
            print(f"  Batch {batch_size:>2}: {throughput:>8.1f} tok/s | {memory:.1f}GB | {elapsed:.2f}s")
            
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {batch_size:>2}: OOM")
            torch.cuda.empty_cache()
            break
    
    del llm
    torch.cuda.empty_cache()
    return results


def main():
    print("=" * 60)
    print("PyTorch vs vLLM Performance Comparison")
    print("=" * 60)
    
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu}")
    print(f"Memory: {mem:.1f} GB")
    
    # Models and batch sizes based on GPU memory
    if mem >= 70:
        models = ["Qwen/Qwen2.5-7B", "microsoft/Phi-3-mini-4k-instruct"]
        batch_sizes = [1, 4, 8, 16, 32, 64]
    elif mem >= 20:
        models = ["Qwen/Qwen2.5-1.5B"]
        batch_sizes = [1, 4, 8, 16, 32]
    else:
        models = ["Qwen/Qwen2.5-0.5B"]
        batch_sizes = [1, 4, 8]
    
    all_results = {}
    
    for model_name in models:
        model_short = model_name.split('/')[-1]
        all_results[model_short] = {}
        
        # PyTorch
        pytorch_results = pytorch_benchmark(model_name, batch_sizes)
        all_results[model_short]['PyTorch'] = pytorch_results
        
        # vLLM
        vllm_results = vllm_benchmark(model_name, batch_sizes)
        all_results[model_short]['vLLM'] = vllm_results
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    for model, results in all_results.items():
        print(f"\n{model}:")
        print(f"{'Framework':<12} {'Best Batch':>10} {'Peak Throughput':>16} {'Memory':>10}")
        print("-" * 50)
        
        pytorch_peak = 0
        vllm_peak = 0
        
        if results.get('PyTorch'):
            best = max(results['PyTorch'], key=lambda x: x['throughput'])
            pytorch_peak = best['throughput']
            print(f"{'PyTorch':<12} {best['batch']:>10} {best['throughput']:>14.1f}/s {best['memory']:>8.1f}GB")
        
        if results.get('vLLM'):
            best = max(results['vLLM'], key=lambda x: x['throughput'])
            vllm_peak = best['throughput']
            print(f"{'vLLM':<12} {best['batch']:>10} {best['throughput']:>14.1f}/s {best['memory']:>8.1f}GB")
        
        if pytorch_peak > 0 and vllm_peak > 0:
            speedup = (vllm_peak / pytorch_peak - 1) * 100
            print(f"\nvLLM speedup over PyTorch: {speedup:+.1f}%")
    
    # Detailed per-batch comparison
    print("\n" + "=" * 70)
    print("PER-BATCH COMPARISON")
    print("=" * 70)
    
    for model, results in all_results.items():
        print(f"\n{model}:")
        print(f"{'Batch':>6} {'PyTorch':>12} {'vLLM':>12} {'Speedup':>10}")
        print("-" * 45)
        
        pytorch_dict = {r['batch']: r['throughput'] for r in results.get('PyTorch', [])}
        vllm_dict = {r['batch']: r['throughput'] for r in results.get('vLLM', [])}
        
        all_batches = sorted(set(pytorch_dict.keys()) | set(vllm_dict.keys()))
        
        for batch in all_batches:
            pt = pytorch_dict.get(batch, 0)
            vl = vllm_dict.get(batch, 0)
            
            pt_str = f"{pt:.1f}" if pt > 0 else "OOM"
            vl_str = f"{vl:.1f}" if vl > 0 else "OOM"
            
            if pt > 0 and vl > 0:
                speedup = (vl / pt - 1) * 100
                sp_str = f"{speedup:+.1f}%"
            else:
                sp_str = "-"
            
            print(f"{batch:>6} {pt_str:>12} {vl_str:>12} {sp_str:>10}")


if __name__ == "__main__":
    main()
EOF

python3 /tmp/vllm_compare.py
