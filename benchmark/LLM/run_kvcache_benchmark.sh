#!/bin/bash
# Script to run the Llama-3 KVCache benchmark and analyze results

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/../../build"
RESULTS_DIR="$SCRIPT_DIR/results"
BENCHMARK_EXEC="$BUILD_DIR/bin/llama3_kvcache_benchmark"
RESULTS_FILE="$RESULTS_DIR/kvcache_benchmark_results.txt"
ANALYZE_SCRIPT="$SCRIPT_DIR/analyze_results.py"

# Print information
echo "===== LLMIR PagedKVCache Benchmark with Llama-3 3B Model ====="
echo "Build directory: $BUILD_DIR"
echo "Results directory: $RESULTS_DIR"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Check if benchmark executable exists
if [ ! -f "$BENCHMARK_EXEC" ]; then
    echo "Error: Benchmark executable not found at $BENCHMARK_EXEC"
    echo "Please build the project first using:"
    echo "  mkdir -p $BUILD_DIR"
    echo "  cd $BUILD_DIR"
    
    # Check if running on macOS with Apple Silicon
    if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        echo "  cmake -G Ninja .. -DLLMIR_ENABLE_METAL=ON"
        METAL_FLAG="-DLLMIR_ENABLE_METAL=ON"
    else
        echo "  cmake -G Ninja .. -DLLMIR_ENABLE_CUDA=ON"
        METAL_FLAG=""
    fi
    
    echo "  ninja llama3_kvcache_benchmark"
    exit 1
fi

# Run the benchmark
echo "Running benchmark..."
"$BENCHMARK_EXEC" --benchmark_out="$RESULTS_FILE" --benchmark_out_format=console

# Analyze the results
echo "Analyzing results..."
if command -v python3 &> /dev/null; then
    # Check if required packages are installed
    if python3 -c "import pandas, matplotlib, seaborn" &> /dev/null; then
        python3 "$ANALYZE_SCRIPT" "$RESULTS_FILE"
        
        # Move generated plots to results directory
        mv *.png "$RESULTS_DIR/"
        mv benchmark_summary.txt "$RESULTS_DIR/"
        
        echo "Analysis complete. Results saved to $RESULTS_DIR"
    else
        echo "Required Python packages not found. Please install them using:"
        echo "  pip install pandas matplotlib seaborn"
        echo "Results saved to $RESULTS_FILE"
    fi
else
    echo "Python 3 not found. Results saved to $RESULTS_FILE"
fi

echo "Done." 