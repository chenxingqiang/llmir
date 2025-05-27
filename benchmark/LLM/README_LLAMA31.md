# Benchmarking Llama-3.1-8B-Instruct with LLMIR Optimizations

This directory contains scripts for benchmarking the [Meta Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model with LLMIR optimizations to measure performance improvements. The benchmark uses vLLM as the serving engine and applies LLMIR's optimizations for attention and KV cache operations.

## Prerequisites

- Python 3.8+
- LLMIR built and installed
- An Apple M3 Mac or NVIDIA GPU (recommended)
- Hugging Face account with access to the Llama-3.1 model

## Setup

1. Clone the repository and build LLMIR following the standard build instructions.

2. Run the setup script to prepare the environment:

```bash
cd benchmark/LLM
chmod +x setup_llama31_benchmark.sh
./setup_llama31_benchmark.sh
```

3. Activate the environment:

```bash
source benchmark/LLM/venv/bin/activate_llmir
```

4. Set your Hugging Face token (required to access the Llama-3.1 model):

```bash
export HUGGINGFACE_TOKEN=your_token_here
```

## Running the Benchmark

The benchmark script runs the model with and without LLMIR optimizations and compares the performance:

```bash
./run_llama31_benchmark.sh
```

### Customizing the Benchmark

You can customize the benchmark with various options:

```bash
./run_llama31_benchmark.sh --help
```

Available options:

- `--model MODEL`: HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)
- `--batch-sizes SIZES`: Comma-separated list of batch sizes to test (default: 1,2,4,8)
- `--seq-lens LENS`: Comma-separated list of sequence lengths to test (default: 128,512,1024,2048)
- `--repetitions N`: Number of repetitions for each configuration (default: 3)
- `--port PORT`: Port for the vLLM server (default: 8000)
- `--results-dir DIR`: Directory to save results (default: benchmark/LLM/results/llama31)

Example with custom options:

```bash
./run_llama31_benchmark.sh \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --batch-sizes 1,2,4 \
  --seq-lens 512,1024 \
  --repetitions 5
```

## Understanding the Results

The benchmark outputs several files and visualizations:

1. `llama31_benchmark_results.csv`: Raw benchmark results
2. `llama31_benchmark_summary.csv`: Summary statistics by configuration
3. `latency_by_batch_size.png`: Plot showing latency across different batch sizes and sequence lengths
4. `throughput_by_batch_size.png`: Plot showing throughput across different batch sizes and sequence lengths
5. `llmir_speedup.png`: Plot showing the speedup from LLMIR optimizations

The terminal output will also display a summary of the speedup achieved by LLMIR optimizations.

## vLLM Integration Details

The integration with vLLM happens through environment variables and custom Python modules:

- `LLMIR_OPTIMIZE=1`: Enables LLMIR optimizations (general flag)
- `LLMIR_KV_CACHE_ENABLE=1`: Enables KV cache optimizations
- `LLMIR_ATTENTION_OPTIMIZE=1`: Enables attention optimizations
- `LLMIR_MODEL_OPTIMIZE=1`: Enables model-wide optimizations (optional)

The integration is implemented via a patch file (`llmir_vllm_integration.patch`) that modifies vLLM's attention and model loading code to use LLMIR optimizations when available.

## Using LLMIR with vLLM Directly

If you want to use LLMIR optimizations with vLLM directly in your own applications:

1. Apply the LLMIR patch to your vLLM installation:

```bash
cd /path/to/vllm
patch -p1 < /path/to/llmir/benchmark/LLM/llmir_vllm_integration.patch
```

2. Set the environment variables before running vLLM:

```bash
export LLMIR_OPTIMIZE=1
export LLMIR_KV_CACHE_ENABLE=1
export LLMIR_ATTENTION_OPTIMIZE=1
vllm serve "meta-llama/Llama-3.1-8B-Instruct"
```

3. Use the vLLM API as usual with LLMIR optimizations applied:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ]
    }'
```

## Troubleshooting

- **Error about LLMIR modules not found**: Make sure LLMIR is built and the environment is activated with `source venv/bin/activate_llmir`
- **Error accessing the model**: Ensure your Hugging Face token is set and has access to the Llama-3.1 model
- **Out of memory errors**: Reduce batch sizes or use a device with more VRAM

## Contributing

Feel free to modify the benchmark scripts to test other models or optimization configurations. If you make improvements, please submit a pull request. 