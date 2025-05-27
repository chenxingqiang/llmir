#!/usr/bin/env python3
"""
Benchmark script for comparing Llama-3.1-8B-Instruct performance with and without LLMIR optimizations
"""

import argparse
import json
import os
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import torch
import requests
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Llama-3.1 with LLMIR optimization")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                      help="Hugging Face model ID")
    parser.add_argument("--batch_sizes", type=str, default="1,4,8,16", 
                      help="Comma-separated list of batch sizes to test")
    parser.add_argument("--seq_lens", type=str, default="128,512,1024,2048",
                      help="Comma-separated list of sequence lengths to test")
    parser.add_argument("--repetitions", type=int, default=5,
                      help="Number of repetitions for each configuration")
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Directory to save benchmark results")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Directory to cache model files")
    parser.add_argument("--prompt_file", type=str, default=None,
                      help="JSON file containing prompts to use (otherwise default prompts will be used)")
    parser.add_argument("--llmir_optimize", action="store_true",
                      help="Whether to apply LLMIR optimization")
    parser.add_argument("--port", type=int, default=8000,
                      help="Port to use for the vLLM server")
    
    return parser.parse_args()

def generate_default_prompts() -> List[str]:
    """Generate a set of default prompts for benchmarking"""
    return [
        "What is the capital of France?",
        "Explain the process of photosynthesis in simple terms.",
        "Write a short poem about the changing seasons.",
        "What are the key differences between machine learning and traditional programming?",
        "Describe the plot of the movie 'Inception' in a few sentences.",
        "What are three healthy breakfast options for someone with limited time?",
        "Explain how a car engine works.",
        "What are the main causes of climate change?",
        "Provide some tips for improving productivity when working from home.",
        "Describe the significance of the Mona Lisa painting in art history."
    ]

def load_prompts(prompt_file: str = None) -> List[str]:
    """Load prompts from file or use defaults"""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
        return prompts
    else:
        return generate_default_prompts()

def start_vllm_server(model_id: str, port: int, llmir_optimize: bool = False) -> subprocess.Popen:
    """Start vLLM server with the specified model"""
    cmd = [
        "vllm", "serve", model_id,
        "--port", str(port),
        "--tensor-parallel-size", "1"  # Adjust based on available GPUs
    ]
    
    # Apply LLMIR optimization if requested
    env = os.environ.copy()
    if llmir_optimize:
        env["LLMIR_OPTIMIZE"] = "1"
        env["LLMIR_KV_CACHE_ENABLE"] = "1"
        env["LLMIR_ATTENTION_OPTIMIZE"] = "1"
        print("Running with LLMIR optimizations enabled")
    else:
        print("Running without LLMIR optimizations")
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    # Start the server process
    server = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for server to start
    time.sleep(30)  # Adjust waiting time as needed
    
    return server

def stop_vllm_server(server_process: subprocess.Popen):
    """Stop the vLLM server process"""
    if server_process:
        server_process.terminate()
        try:
            server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("vLLM server stopped")

def run_benchmark(
    model_id: str,
    prompts: List[str],
    batch_sizes: List[int],
    seq_lens: List[int],
    repetitions: int,
    port: int,
    llmir_optimize: bool = False
) -> pd.DataFrame:
    """Run benchmark with different batch sizes and sequence lengths"""
    results = []
    
    # Start the server
    server = start_vllm_server(model_id, port, llmir_optimize)
    
    try:
        base_url = f"http://localhost:{port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        for batch_size in batch_sizes:
            for max_tokens in seq_lens:
                print(f"Testing batch_size={batch_size}, max_tokens={max_tokens}")
                
                # Use a subset of prompts for each batch
                batch_prompts = prompts[:batch_size] if batch_size <= len(prompts) else prompts * (batch_size // len(prompts) + 1)
                batch_prompts = batch_prompts[:batch_size]  # Ensure we have exactly batch_size prompts
                
                # Create messages for each prompt
                messages_list = []
                for prompt in batch_prompts:
                    messages_list.append([{"role": "user", "content": prompt}])
                
                # Run multiple repetitions
                latencies = []
                for rep in range(repetitions):
                    # Create the request payload
                    payload = {
                        "model": model_id,
                        "messages": messages_list[0],  # Single request for simplicity
                        "max_tokens": max_tokens,
                        "temperature": 0.0  # Deterministic for benchmarking
                    }
                    
                    # Measure latency
                    start_time = time.time()
                    response = requests.post(base_url, headers=headers, json=payload)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        latency = (end_time - start_time) * 1000  # ms
                        latencies.append(latency)
                        
                        # Add to results
                        results.append({
                            "optimization": "LLMIR" if llmir_optimize else "Baseline",
                            "batch_size": batch_size,
                            "max_tokens": max_tokens,
                            "repetition": rep,
                            "latency_ms": latency,
                            "throughput_tokens_per_sec": (max_tokens / (latency / 1000))
                        })
                    else:
                        print(f"Error: {response.status_code}, {response.text}")
                
                # Log results for this configuration
                avg_latency = np.mean(latencies)
                std_latency = np.std(latencies)
                print(f"  Avg latency: {avg_latency:.2f} ms ± {std_latency:.2f} ms")
                print(f"  Throughput: {(max_tokens / (avg_latency / 1000)):.2f} tokens/sec")
                
    finally:
        # Always stop the server
        stop_vllm_server(server)
    
    return pd.DataFrame(results)

def save_and_visualize_results(df: pd.DataFrame, output_dir: str, model_id: str):
    """Save benchmark results to CSV and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    result_file = os.path.join(output_dir, "llama31_benchmark_results.csv")
    df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
    
    # Create summary dataframe
    summary_df = df.groupby(['optimization', 'batch_size', 'max_tokens'])[['latency_ms', 'throughput_tokens_per_sec']].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    
    # Flatten multi-level columns
    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
    
    # Save summary
    summary_file = os.path.join(output_dir, "llama31_benchmark_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    
    # Create visualizations
    
    # 1. Latency by batch size and sequence length
    plt.figure(figsize=(12, 8))
    for opt in df['optimization'].unique():
        for max_tokens in df['max_tokens'].unique():
            subset = df[(df['optimization'] == opt) & (df['max_tokens'] == max_tokens)]
            plt.errorbar(
                subset['batch_size'].unique(), 
                subset.groupby('batch_size')['latency_ms'].mean(),
                yerr=subset.groupby('batch_size')['latency_ms'].std(),
                label=f"{opt}, seq_len={max_tokens}",
                marker='o'
            )
    
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (ms)')
    plt.title(f'Inference Latency - {model_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "latency_by_batch_size.png"))
    
    # 2. Throughput by batch size and sequence length
    plt.figure(figsize=(12, 8))
    for opt in df['optimization'].unique():
        for max_tokens in df['max_tokens'].unique():
            subset = df[(df['optimization'] == opt) & (df['max_tokens'] == max_tokens)]
            plt.errorbar(
                subset['batch_size'].unique(), 
                subset.groupby('batch_size')['throughput_tokens_per_sec'].mean(),
                yerr=subset.groupby('batch_size')['throughput_tokens_per_sec'].std(),
                label=f"{opt}, seq_len={max_tokens}",
                marker='o'
            )
    
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title(f'Inference Throughput - {model_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "throughput_by_batch_size.png"))
    
    # 3. Speedup comparison (if both baseline and LLMIR are present)
    if len(df['optimization'].unique()) > 1:
        plt.figure(figsize=(12, 8))
        
        # Calculate speedup for each batch size and sequence length
        speedup_data = []
        
        for batch_size in df['batch_size'].unique():
            for max_tokens in df['max_tokens'].unique():
                baseline = df[(df['optimization'] == 'Baseline') & 
                             (df['batch_size'] == batch_size) & 
                             (df['max_tokens'] == max_tokens)]['latency_ms'].mean()
                
                llmir = df[(df['optimization'] == 'LLMIR') & 
                          (df['batch_size'] == batch_size) & 
                          (df['max_tokens'] == max_tokens)]['latency_ms'].mean()
                
                if baseline > 0 and llmir > 0:
                    speedup_data.append({
                        'batch_size': batch_size,
                        'max_tokens': max_tokens,
                        'speedup': baseline / llmir
                    })
        
        speedup_df = pd.DataFrame(speedup_data)
        
        for max_tokens in speedup_df['max_tokens'].unique():
            subset = speedup_df[speedup_df['max_tokens'] == max_tokens]
            plt.plot(
                subset['batch_size'],
                subset['speedup'],
                label=f"seq_len={max_tokens}",
                marker='o'
            )
        
        plt.axhline(y=1.0, color='r', linestyle='--', label='No speedup')
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup (Baseline latency / LLMIR latency)')
        plt.title(f'LLMIR Optimization Speedup - {model_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "llmir_speedup.png"))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    args = parse_args()
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    seq_lens = [int(sl) for sl in args.seq_lens.split(',')]
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmark without LLMIR optimization (baseline)
    print("Running baseline benchmark...")
    baseline_results = run_benchmark(
        args.model,
        prompts,
        batch_sizes,
        seq_lens,
        args.repetitions,
        args.port
    )
    
    # Run benchmark with LLMIR optimization
    print("Running LLMIR-optimized benchmark...")
    llmir_results = run_benchmark(
        args.model,
        prompts,
        batch_sizes,
        seq_lens,
        args.repetitions,
        args.port,
        llmir_optimize=True
    )
    
    # Combine results
    all_results = pd.concat([baseline_results, llmir_results])
    
    # Save and visualize results
    save_and_visualize_results(all_results, args.output_dir, args.model)
    
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main() 