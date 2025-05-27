#!/bin/bash
# Setup script for Llama-3.1 benchmark with LLMIR

set -e  # Exit on error

# Directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/../.."
VENV_DIR="$SCRIPT_DIR/venv"

# Print header
echo "===== Setting up Llama-3.1 Benchmark Environment ====="
echo "Repository root: $REPO_ROOT"
echo "Virtual environment: $VENV_DIR"
echo "======================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"

# Determine the activate script based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Mac OS detected"
    # Check for Metal support
    if [[ $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
        echo "Apple Silicon detected, will use MPS backend"
        TORCH_EXTRA="--index-url https://download.pytorch.org/whl/nightly/cpu"
    else
        echo "Intel Mac detected"
        TORCH_EXTRA=""
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Linux detected"
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, will use CUDA backend"
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d "." -f 1)
        if [ "$CUDA_MAJOR" -ge 11 ]; then
            TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cu118"
        else
            TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cu117"
        fi
        echo "CUDA version: $CUDA_VERSION"
    else
        echo "No NVIDIA GPU detected, will use CPU backend"
        TORCH_EXTRA="--index-url https://download.pytorch.org/whl/cpu"
    fi
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Install PyTorch
echo "Installing PyTorch..."
pip install --upgrade pip
pip install torch torchvision torchaudio $TORCH_EXTRA

# Install vLLM
echo "Installing vLLM..."
pip install vllm

# Install other required packages
echo "Installing other required packages..."
pip install pandas matplotlib seaborn numpy requests tqdm huggingface_hub

# Check LLMIR availability
echo "Checking LLMIR availability..."
if [ -d "$REPO_ROOT/lib/Dialect/LLM" ]; then
    echo "LLMIR found at $REPO_ROOT/lib/Dialect/LLM"
    
    # Set up environment variables for LLMIR
    echo "Setting up LLMIR environment..."
    
    # Create or update .env file
    ENV_FILE="$SCRIPT_DIR/.env"
    echo "# LLMIR Environment Variables" > "$ENV_FILE"
    echo "LLMIR_ROOT=$REPO_ROOT" >> "$ENV_FILE"
    echo "LLMIR_PYTHON_PATH=$REPO_ROOT/python" >> "$ENV_FILE"
    echo "LLMIR_LIB_PATH=$REPO_ROOT/build/lib" >> "$ENV_FILE"
    echo "PYTHONPATH=\$PYTHONPATH:$REPO_ROOT/python" >> "$ENV_FILE"
    echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$REPO_ROOT/build/lib" >> "$ENV_FILE"
    echo "DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$REPO_ROOT/build/lib" >> "$ENV_FILE"
    
    # Create activation script
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate_llmir"
    echo "# LLMIR Activation Script" > "$ACTIVATE_SCRIPT"
    echo "source $VENV_DIR/bin/activate" >> "$ACTIVATE_SCRIPT"
    echo "export LLMIR_ROOT=$REPO_ROOT" >> "$ACTIVATE_SCRIPT"
    echo "export LLMIR_PYTHON_PATH=$REPO_ROOT/python" >> "$ACTIVATE_SCRIPT"
    echo "export LLMIR_LIB_PATH=$REPO_ROOT/build/lib" >> "$ACTIVATE_SCRIPT"
    echo "export PYTHONPATH=\$PYTHONPATH:$REPO_ROOT/python" >> "$ACTIVATE_SCRIPT"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$REPO_ROOT/build/lib" >> "$ACTIVATE_SCRIPT"
    echo "export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$REPO_ROOT/build/lib" >> "$ACTIVATE_SCRIPT"
    chmod +x "$ACTIVATE_SCRIPT"
    
    echo "Created LLMIR activation script at $ACTIVATE_SCRIPT"
    echo "To activate the environment with LLMIR support, run:"
    echo "source $ACTIVATE_SCRIPT"
else
    echo "Warning: LLMIR not found at $REPO_ROOT/lib/Dialect/LLM"
    echo "Make sure to build LLMIR before running the benchmark"
fi

# Make benchmark scripts executable
chmod +x "$SCRIPT_DIR/run_llama31_benchmark.sh"
chmod +x "$SCRIPT_DIR/llama31_benchmark.py"

# Print instructions
echo 
echo "===== Setup Complete ====="
echo "To run the benchmark:"
echo "1. Activate the environment: source $VENV_DIR/bin/activate"
echo "2. Set your HuggingFace token (if needed): export HUGGINGFACE_TOKEN=your_token"
echo "3. Run the benchmark: $SCRIPT_DIR/run_llama31_benchmark.sh"
echo
echo "You can customize the benchmark with options:"
echo "$SCRIPT_DIR/run_llama31_benchmark.sh --help"
echo "===== Happy Benchmarking! =====" 