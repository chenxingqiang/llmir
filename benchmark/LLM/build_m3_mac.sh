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
cd "$BUILD_DIR" || exit 1

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Please install cmake first."
    exit 1
fi

if ! command -v ninja &> /dev/null; then
    echo "Error: ninja not found. Please install ninja first."
    exit 1
fi

# Configure with Metal support
echo "Configuring with Metal support..."
cmake -G Ninja .. -DLLMIR_ENABLE_METAL=ON

# Build
echo "Building benchmark..."
ninja llama3_kvcache_benchmark

echo "Done. Run the benchmark with ./benchmark/LLM/run_kvcache_benchmark.sh" 