#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="$SCRIPT_DIR/../../../../build"

# Make sure we're in the build directory
cd $BUILD_DIR

# Build the KV cache unit tests
echo "Building KV cache unit tests..."
cmake --build . --target LLMRuntimeTests

# Run the tests
echo "Running KV cache unit tests..."
./bin/LLMRuntimeTests --gtest_filter="*KVCache*"

echo "Tests completed successfully!" 