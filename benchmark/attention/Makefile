CXX = g++
CXXFLAGS = -std=c++17 -O3

BENCHMARKS = simple_attention_benchmark simple_mqa_benchmark simple_pruned_attention_benchmark

all: $(BENCHMARKS)

simple_attention_benchmark: simple_attention_benchmark.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

simple_mqa_benchmark: simple_mqa_benchmark.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

simple_pruned_attention_benchmark: simple_pruned_attention_benchmark.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(BENCHMARKS)

run: all
	./run_benchmarks.sh

results: run
	python3 parse_results.py

.PHONY: all clean run results 