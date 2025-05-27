#!/bin/bash
# Script to build the KVCache benchmark for Apple M3 Mac

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/../../build"

# Print information
echo "===== Building LLMIR PagedKVCache Benchmark for Apple M3 Mac ====="
echo "Source directory: $SCRIPT_DIR/../../"
echo "Build directory: $BUILD_DIR"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"
mkdir -p "$BUILD_DIR/bin"

# Check for required tools
if ! command -v clang++ &> /dev/null; then
    echo "Error: clang++ not found. Please install Xcode command line tools."
    exit 1
fi

# Check for benchmark library
if ! brew list --versions google-benchmark &> /dev/null; then
    echo "Google Benchmark not found. Installing..."
    brew install google-benchmark
fi

# Get library paths
BENCHMARK_DIR=$(brew --prefix google-benchmark)
echo "Found Google Benchmark at: $BENCHMARK_DIR"

# Compile the benchmark
echo "Compiling benchmark for Apple M3 Mac with Metal..."

clang++ -std=c++17 -O3 \
  -DLLMIR_ENABLE_METAL \
  -framework Metal -framework Foundation \
  -I$BENCHMARK_DIR/include \
  -L$BENCHMARK_DIR/lib \
  -lbenchmark -lbenchmark_main \
  $SCRIPT_DIR/simple_mac_benchmark.mm \
  -o $BUILD_DIR/bin/llama3_kvcache_benchmark

# Check if build was successful
if [ $? -eq 0 ] && [ -f "$BUILD_DIR/bin/llama3_kvcache_benchmark" ]; then
    echo "Benchmark compiled successfully!"
    chmod +x $BUILD_DIR/bin/llama3_kvcache_benchmark
else
    echo "Error: Failed to compile benchmark"
    exit 1
fi

echo "Done. Run the benchmark with ./benchmark/LLM/run_kvcache_benchmark.sh" 