#!/bin/bash
# Script to run the Llama-3.1-8B-Instruct benchmark with LLMIR optimizations

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="$SCRIPT_DIR/results/llama31"
mkdir -p "$RESULTS_DIR"

# Check for Python and required packages
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check for vLLM
if ! command -v vllm &> /dev/null; then
    echo "Error: vLLM is not installed. Please install it with:"
    echo "pip install vllm"
    exit 1
fi

# Check if LLMIR is available in the environment
if ! python3 -c "import torch; import os; print('LLMIR_AVAILABLE=1' if os.path.exists('$SCRIPT_DIR/../../lib/Dialect/LLM') else 'LLMIR_AVAILABLE=0')" | grep -q "LLMIR_AVAILABLE=1"; then
    echo "Warning: LLMIR doesn't seem to be available or properly installed."
    echo "Make sure LLMIR is built and installed correctly."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default benchmark parameters
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZES="1,2,4,8"
SEQ_LENS="128,512,1024,2048"
REPETITIONS=3
PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --seq-lens)
            SEQ_LENS="$2"
            shift 2
            ;;
        --repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL        HuggingFace model ID (default: $MODEL)"
            echo "  --batch-sizes SIZES  Comma-separated list of batch sizes to test (default: $BATCH_SIZES)"
            echo "  --seq-lens LENS      Comma-separated list of sequence lengths to test (default: $SEQ_LENS)"
            echo "  --repetitions N      Number of repetitions for each configuration (default: $REPETITIONS)"
            echo "  --port PORT          Port for the vLLM server (default: $PORT)"
            echo "  --results-dir DIR    Directory to save results (default: $RESULTS_DIR)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if HuggingFace token is available (needed for some models)
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Warning: HUGGINGFACE_TOKEN environment variable is not set."
    echo "Some models may require authentication. If you have a token,"
    echo "please set it with: export HUGGINGFACE_TOKEN=your_token"
fi

# Install required Python packages if not already installed
echo "Checking Python dependencies..."
python3 -m pip install --quiet pandas matplotlib seaborn numpy requests torch tqdm

# Print configuration
echo "===== LLMIR Benchmark for Llama-3.1 ====="
echo "Model: $MODEL"
echo "Batch sizes: $BATCH_SIZES"
echo "Sequence lengths: $SEQ_LENS"
echo "Repetitions: $REPETITIONS"
echo "Results directory: $RESULTS_DIR"
echo "=======================================\n"

# Run the benchmark script
echo "Starting benchmark..."
python3 "$SCRIPT_DIR/llama31_benchmark.py" \
    --model "$MODEL" \
    --batch_sizes "$BATCH_SIZES" \
    --seq_lens "$SEQ_LENS" \
    --repetitions "$REPETITIONS" \
    --port "$PORT" \
    --output_dir "$RESULTS_DIR"

# Display results
if [ -f "$RESULTS_DIR/llama31_benchmark_summary.csv" ]; then
    echo "\nBenchmark Summary:"
    cat "$RESULTS_DIR/llama31_benchmark_summary.csv"
    
    echo "\nSpeedup in LLMIR-optimized version:"
    python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('$RESULTS_DIR/llama31_benchmark_results.csv')
speedups = []

for batch in sorted(df['batch_size'].unique()):
    for seq_len in sorted(df['max_tokens'].unique()):
        baseline = df[(df['optimization'] == 'Baseline') & 
                      (df['batch_size'] == batch) & 
                      (df['max_tokens'] == seq_len)]['latency_ms'].mean()
        
        llmir = df[(df['optimization'] == 'LLMIR') & 
                   (df['batch_size'] == batch) & 
                   (df['max_tokens'] == seq_len)]['latency_ms'].mean()
        
        if baseline > 0 and llmir > 0:
            speedup = baseline / llmir
            speedups.append(speedup)
            print(f'Batch size {batch}, Seq length {seq_len}: {speedup:.2f}x speedup')

if speedups:
    print(f'\nAverage speedup: {np.mean(speedups):.2f}x')
    print(f'Maximum speedup: {np.max(speedups):.2f}x')
else:
    print('No valid speedup measurements found.')
"
    
    # Open plots on macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "\nOpening result plots..."
        open "$RESULTS_DIR/latency_by_batch_size.png"
        open "$RESULTS_DIR/throughput_by_batch_size.png"
        open "$RESULTS_DIR/llmir_speedup.png"
    else
        echo "\nResults and plots saved to: $RESULTS_DIR"
    fi
else
    echo "\nError: Benchmark did not complete successfully. Check logs for details."
fi

echo "\nBenchmark completed." 