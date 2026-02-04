#!/bin/bash
# Comprehensive LLM Inference Benchmark
# 对比 PyTorch, vLLM, SGLang 三种框架的性能

set -e

echo "=============================================="
echo "Comprehensive LLM Inference Benchmark"
echo "PyTorch vs vLLM vs SGLang"
echo "=============================================="

# 设置环境
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface

# 检查GPU
echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# 安装依赖
echo "=== Installing Dependencies ==="
pip install torch transformers accelerate sentencepiece tiktoken -q
pip install vllm -q
pip install "sglang[all]" -q
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ -q 2>/dev/null || true

echo ""
echo "=== Package Versions ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: not installed"
python3 -c "import sglang; print(f'SGLang: {sglang.__version__}')" 2>/dev/null || echo "SGLang: not installed"

# 创建综合测试脚本
cat > /tmp/comprehensive_benchmark.py << 'BENCHMARK_EOF'
#!/usr/bin/env python3
"""
Comprehensive LLM Inference Benchmark
Compare PyTorch, vLLM, and SGLang performance
"""

import os
import sys
import time
import json
import torch
import gc
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

@dataclass
class BenchmarkResult:
    framework: str
    model: str
    batch_size: int
    input_len: int
    output_len: int
    time_s: float
    throughput: float  # tokens/s
    latency_ms: float  # ms per token
    memory_gb: float
    success: bool = True
    error: str = ""

class PyTorchBenchmark:
    """PyTorch (transformers) baseline benchmark"""
    
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"  Loading PyTorch model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_name = model_name
    
    def benchmark(self, batch_size: int, input_len: int, output_len: int) -> BenchmarkResult:
        prompt = "Explain machine learning in detail: " * (input_len // 5)
        inputs = self.tokenizer([prompt] * batch_size, return_tensors="pt",
                               truncation=True, max_length=input_len, padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        actual_input_len = inputs['input_ids'].shape[1]
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate(**inputs, max_new_tokens=5, 
                                   pad_token_id=self.tokenizer.pad_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=output_len,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        generated = outputs.shape[1] - actual_input_len
        total_tokens = batch_size * generated
        throughput = total_tokens / elapsed
        latency = elapsed / generated * 1000
        memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        return BenchmarkResult(
            framework="PyTorch",
            model=self.model_name.split('/')[-1],
            batch_size=batch_size,
            input_len=actual_input_len,
            output_len=generated,
            time_s=elapsed,
            throughput=throughput,
            latency_ms=latency,
            memory_gb=memory,
        )
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()


class VLLMBenchmark:
    """vLLM benchmark"""
    
    def __init__(self, model_name: str):
        from vllm import LLM, SamplingParams
        
        print(f"  Loading vLLM model: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.85,
        )
        self.model_name = model_name
        self.SamplingParams = SamplingParams
    
    def benchmark(self, batch_size: int, input_len: int, output_len: int) -> BenchmarkResult:
        prompt = "Explain machine learning in detail: " * (input_len // 5)
        prompts = [prompt] * batch_size
        
        sampling_params = self.SamplingParams(
            max_tokens=output_len,
            temperature=0,
        )
        
        # Warmup
        _ = self.llm.generate(prompts[:1], self.SamplingParams(max_tokens=5, temperature=0))
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed
        latency = elapsed / (total_tokens / batch_size) * 1000
        memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        return BenchmarkResult(
            framework="vLLM",
            model=self.model_name.split('/')[-1],
            batch_size=batch_size,
            input_len=input_len,
            output_len=total_tokens // batch_size,
            time_s=elapsed,
            throughput=throughput,
            latency_ms=latency,
            memory_gb=memory,
        )
    
    def cleanup(self):
        del self.llm
        torch.cuda.empty_cache()
        gc.collect()


class SGLangBenchmark:
    """SGLang benchmark"""
    
    def __init__(self, model_name: str):
        print(f"  Loading SGLang model: {model_name}")
        self.model_name = model_name
        self.engine = None
        
        try:
            import sglang as sgl
            self.sgl = sgl
            
            # SGLang uses a runtime that needs to be started
            self.runtime = sgl.Runtime(
                model_path=model_name,
                trust_remote_code=True,
            )
            sgl.set_default_backend(self.runtime)
            self.available = True
        except Exception as e:
            print(f"  SGLang init error: {e}")
            self.available = False
    
    def benchmark(self, batch_size: int, input_len: int, output_len: int) -> BenchmarkResult:
        if not self.available:
            return BenchmarkResult(
                framework="SGLang",
                model=self.model_name.split('/')[-1],
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
                time_s=0,
                throughput=0,
                latency_ms=0,
                memory_gb=0,
                success=False,
                error="SGLang not available"
            )
        
        prompt = "Explain machine learning in detail: " * (input_len // 5)
        
        @self.sgl.function
        def generate(s):
            s += prompt
            s += self.sgl.gen("response", max_tokens=output_len, temperature=0)
        
        # Warmup
        _ = generate.run()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start = time.perf_counter()
        states = generate.run_batch([{} for _ in range(batch_size)])
        elapsed = time.perf_counter() - start
        
        total_tokens = sum(len(s["response"].split()) for s in states) * 1.3  # rough token estimate
        throughput = batch_size * output_len / elapsed  # use requested output_len
        latency = elapsed / output_len * 1000
        memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        return BenchmarkResult(
            framework="SGLang",
            model=self.model_name.split('/')[-1],
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            time_s=elapsed,
            throughput=throughput,
            latency_ms=latency,
            memory_gb=memory,
        )
    
    def cleanup(self):
        if self.available and hasattr(self, 'runtime'):
            self.runtime.shutdown()
        torch.cuda.empty_cache()
        gc.collect()


def run_benchmark_suite(model_name: str, configs: List[tuple]) -> List[BenchmarkResult]:
    """Run benchmark for all frameworks on a model"""
    results = []
    
    # PyTorch benchmark
    print("\n[PyTorch Benchmark]")
    try:
        pytorch_bench = PyTorchBenchmark(model_name)
        for batch_size, input_len, output_len in configs:
            try:
                result = pytorch_bench.benchmark(batch_size, input_len, output_len)
                results.append(result)
                print(f"  batch={batch_size}: {result.throughput:.1f} tok/s, {result.memory_gb:.1f}GB")
            except torch.cuda.OutOfMemoryError:
                print(f"  batch={batch_size}: OOM")
                results.append(BenchmarkResult(
                    framework="PyTorch", model=model_name.split('/')[-1],
                    batch_size=batch_size, input_len=input_len, output_len=output_len,
                    time_s=0, throughput=0, latency_ms=0, memory_gb=0,
                    success=False, error="OOM"
                ))
                torch.cuda.empty_cache()
        pytorch_bench.cleanup()
    except Exception as e:
        print(f"  PyTorch error: {e}")
    
    # vLLM benchmark
    print("\n[vLLM Benchmark]")
    try:
        vllm_bench = VLLMBenchmark(model_name)
        for batch_size, input_len, output_len in configs:
            try:
                result = vllm_bench.benchmark(batch_size, input_len, output_len)
                results.append(result)
                print(f"  batch={batch_size}: {result.throughput:.1f} tok/s, {result.memory_gb:.1f}GB")
            except torch.cuda.OutOfMemoryError:
                print(f"  batch={batch_size}: OOM")
                results.append(BenchmarkResult(
                    framework="vLLM", model=model_name.split('/')[-1],
                    batch_size=batch_size, input_len=input_len, output_len=output_len,
                    time_s=0, throughput=0, latency_ms=0, memory_gb=0,
                    success=False, error="OOM"
                ))
                torch.cuda.empty_cache()
        vllm_bench.cleanup()
    except Exception as e:
        print(f"  vLLM error: {e}")
    
    # SGLang benchmark
    print("\n[SGLang Benchmark]")
    try:
        sglang_bench = SGLangBenchmark(model_name)
        for batch_size, input_len, output_len in configs:
            try:
                result = sglang_bench.benchmark(batch_size, input_len, output_len)
                results.append(result)
                if result.success:
                    print(f"  batch={batch_size}: {result.throughput:.1f} tok/s, {result.memory_gb:.1f}GB")
                else:
                    print(f"  batch={batch_size}: {result.error}")
            except torch.cuda.OutOfMemoryError:
                print(f"  batch={batch_size}: OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  batch={batch_size}: {e}")
        sglang_bench.cleanup()
    except Exception as e:
        print(f"  SGLang error: {e}")
    
    return results


def main():
    print("=" * 70)
    print("Comprehensive LLM Inference Benchmark")
    print("PyTorch (transformers) vs vLLM vs SGLang")
    print("=" * 70)
    
    # GPU Info
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nGPU: {gpu_name}")
    print(f"Memory: {gpu_mem:.1f} GB")
    
    all_results = []
    
    # Test configurations: (batch_size, input_len, output_len)
    configs = [
        (1, 512, 128),
        (4, 512, 128),
        (8, 512, 128),
        (16, 512, 128),
        (32, 512, 128),
    ]
    
    # Models to test (based on available GPU memory)
    models = []
    if gpu_mem >= 70:
        models = [
            "Qwen/Qwen2.5-7B",
            "microsoft/Phi-3-mini-4k-instruct",
        ]
    elif gpu_mem >= 20:
        models = [
            "Qwen/Qwen2.5-1.5B",
            "microsoft/Phi-3-mini-4k-instruct",
        ]
    else:
        models = ["Qwen/Qwen2.5-0.5B"]
    
    # Run benchmarks
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        
        results = run_benchmark_suite(model_name, configs)
        all_results.extend(results)
    
    # Print summary table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    
    # Group by model
    models_tested = list(set(r.model for r in all_results if r.success))
    
    for model in models_tested:
        print(f"\n--- {model} ---")
        print(f"{'Framework':<12} {'Batch':>6} {'Input':>6} {'Output':>7} {'Throughput':>12} {'Latency':>10} {'Memory':>8}")
        print("-" * 70)
        
        model_results = [r for r in all_results if r.model == model and r.success]
        model_results.sort(key=lambda x: (x.framework, x.batch_size))
        
        for r in model_results:
            print(f"{r.framework:<12} {r.batch_size:>6} {r.input_len:>6} {r.output_len:>7} "
                  f"{r.throughput:>10.1f}/s {r.latency_ms:>8.2f}ms {r.memory_gb:>6.1f}GB")
    
    # Peak throughput comparison
    print("\n" + "=" * 90)
    print("PEAK THROUGHPUT COMPARISON")
    print("=" * 90)
    
    for model in models_tested:
        print(f"\n{model}:")
        model_results = [r for r in all_results if r.model == model and r.success]
        
        for framework in ["PyTorch", "vLLM", "SGLang"]:
            fw_results = [r for r in model_results if r.framework == framework]
            if fw_results:
                best = max(fw_results, key=lambda x: x.throughput)
                print(f"  {framework:<10}: {best.throughput:>10.1f} tokens/s (batch={best.batch_size})")
            else:
                print(f"  {framework:<10}: N/A")
        
        # Calculate speedup
        pytorch_results = [r for r in model_results if r.framework == "PyTorch"]
        vllm_results = [r for r in model_results if r.framework == "vLLM"]
        sglang_results = [r for r in model_results if r.framework == "SGLang"]
        
        if pytorch_results and vllm_results:
            pytorch_best = max(pytorch_results, key=lambda x: x.throughput).throughput
            vllm_best = max(vllm_results, key=lambda x: x.throughput).throughput
            speedup = (vllm_best / pytorch_best - 1) * 100
            print(f"  vLLM speedup over PyTorch: {speedup:+.1f}%")
        
        if pytorch_results and sglang_results:
            pytorch_best = max(pytorch_results, key=lambda x: x.throughput).throughput
            sglang_best = max(sglang_results, key=lambda x: x.throughput).throughput
            speedup = (sglang_best / pytorch_best - 1) * 100
            print(f"  SGLang speedup over PyTorch: {speedup:+.1f}%")
    
    # Save results to JSON
    results_dict = [asdict(r) for r in all_results]
    with open('/tmp/benchmark_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to /tmp/benchmark_results.json")


if __name__ == "__main__":
    main()
BENCHMARK_EOF

# 运行测试
echo ""
echo "=== Starting Comprehensive Benchmark ==="
python3 /tmp/comprehensive_benchmark.py

echo ""
echo "=== Benchmark Complete ==="
