#!/bin/bash
# Script to run the Llama-3 KVCache benchmark and analyze results

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/../../build"
RESULTS_DIR="$SCRIPT_DIR/results"
BENCHMARK_EXEC="$BUILD_DIR/bin/llama3_kvcache_benchmark"
RESULTS_FILE="$RESULTS_DIR/kvcache_benchmark_results.txt"
ANALYZE_SCRIPT="$SCRIPT_DIR/analyze_m3_results.py"

# Print information
echo "===== LLMIR PagedKVCache Benchmark with Llama-3 3B Model ====="
echo "Build directory: $BUILD_DIR"
echo "Results directory: $RESULTS_DIR"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Check if using macOS with Apple Silicon
IS_MAC_SILICON=false
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    IS_MAC_SILICON=true
    echo "Detected Apple Silicon Mac environment"
fi

# Check if benchmark executable exists
if [ ! -f "$BENCHMARK_EXEC" ]; then
    echo "Error: Benchmark executable not found at $BENCHMARK_EXEC"
    echo "Please build the project first using:"
    
    if $IS_MAC_SILICON; then
        echo "  ./benchmark/LLM/build_m3_mac.sh"
    else
        echo "  mkdir -p $BUILD_DIR"
        echo "  cd $BUILD_DIR"
        echo "  cmake -G Ninja .. -DLLMIR_ENABLE_CUDA=ON"
        echo "  ninja llama3_kvcache_benchmark"
    fi
    
    exit 1
fi

# Run the benchmark
echo "Running benchmark..."

# Run with output to console and file
"$BENCHMARK_EXEC" --benchmark_out="$RESULTS_FILE" --benchmark_out_format=console

# Check if run was successful
if [ $? -ne 0 ]; then
    echo "Error: Benchmark execution failed"
    exit 1
fi

echo "Benchmark completed. Results saved to $RESULTS_FILE"

# Parse and display results summary
echo "Results summary:"
echo "----------------------------------------"
grep -A 1 "^KVCache.*mean" "$RESULTS_FILE" | grep -v "^--"
echo "----------------------------------------"

# Generate plots if Python environment available
if command -v python3 &> /dev/null; then
    if [ -f "$ANALYZE_SCRIPT" ]; then
        echo "Analyzing results with Python script..."
        if python3 -c "import pandas, matplotlib, seaborn" &> /dev/null; then
            python3 "$ANALYZE_SCRIPT" "$RESULTS_FILE"
            
            # Move generated plots to results directory
            if [ -f "block_size_performance.png" ]; then
                mv -f block_size_performance.png "$RESULTS_DIR/"
            fi
            if [ -f "batch_seq_performance.png" ]; then
                mv -f batch_seq_performance.png "$RESULTS_DIR/"
            fi
            if [ -f "config_performance.png" ]; then
                mv -f config_performance.png "$RESULTS_DIR/"
            fi
            if [ -f "benchmark_summary.txt" ]; then
                mv -f benchmark_summary.txt "$RESULTS_DIR/"
            fi
            
            echo "Analysis complete. Results and plots saved to $RESULTS_DIR"
            
            # Open plots on macOS
            if [[ "$(uname)" == "Darwin" ]]; then
                echo "Opening plots..."
                open "$RESULTS_DIR/block_size_performance.png"
                open "$RESULTS_DIR/batch_seq_performance.png"
                open "$RESULTS_DIR/config_performance.png"
                
                # Also open the summary text file
                open "$RESULTS_DIR/benchmark_summary.txt"
            fi
        else
            echo "Required Python packages not found. Please install them using:"
            echo "  pip install pandas matplotlib seaborn"
        fi
    else
        echo "Warning: Analysis script not found at $ANALYZE_SCRIPT"
    fi
else
    echo "Python 3 not found. Skipping result analysis."
fi

echo "Done." 