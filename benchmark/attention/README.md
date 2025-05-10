# Attention Optimization Benchmarks

This directory contains benchmarks for various attention optimization techniques implemented in the LLMIR project.

## Included Benchmarks

1. **Flash Attention** (`simple_attention_benchmark.cpp`) - Compares Flash Attention to standard attention
2. **Multi-Query Attention** (`simple_mqa_benchmark.cpp`) - Compares Multi-Query Attention to standard Multi-Head Attention
3. **Pruned Attention** (`simple_pruned_attention_benchmark.cpp`) - Compares threshold-based and top-K pruning to standard attention

## Building the Benchmarks

### Using Make

```bash
make
```

### Using CMake

```bash
mkdir -p ../../build
cd ../../build
cmake ..
cmake --build .
```

## Running the Benchmarks

### Running All Benchmarks

```bash
make run
```

or

```bash
./run_benchmarks.sh
```

### Running Individual Benchmarks

```bash
# Flash Attention
./simple_attention_benchmark --seq 512 --context 512 --trials 10

# Multi-Query Attention
./simple_mqa_benchmark --seq 512 --context 512 --trials 10

# Pruned Attention
./simple_pruned_attention_benchmark --seq 512 --context 512 --trials 10
```

## Benchmark Parameters

All benchmarks support the following common parameters:

- `--seq <value>`: Sequence length for queries (default: 512)
- `--context <value>`: Context length for keys/values (default: 512)
- `--batch <value>`: Batch size (default: 2)
- `--heads <value>`: Number of attention heads (default: 12)
- `--dim <value>`: Head dimension (default: 64)
- `--trials <value>`: Number of trials to run (default: 5)

Pruned attention benchmark also supports:
- `--threshold <value>`: Threshold for pruning (default: 0.01)
- `--topk <value>`: K value for Top-K pruning (default: 128)

## Generating Results and Plots

After running the benchmarks:

```bash
make results
```

or

```bash
python3 parse_results.py
```

This will:
1. Generate plots in the `plots/` directory
2. Create a markdown report at `../docs/attention_optimization_test_report.md`

## Directory Structure

- `*.cpp`: Benchmark source files
- `results/`: Raw benchmark results
- `plots/`: Generated performance visualizations
- `run_benchmarks.sh`: Script to run all benchmarks
- `parse_results.py`: Script to generate visualizations and reports 