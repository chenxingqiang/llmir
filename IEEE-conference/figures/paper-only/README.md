# Paper figure generators (illustrative data)

Python scripts in this directory build **IEEE submission figures** using
**hard-coded or hand-tuned arrays**. They are **not** wired to automated
benchmark output from `llmir-benchmark` or CI.

For reproducible measurements use:

- [`scripts/cpu_inference_compare.py`](../../../scripts/cpu_inference_compare.py)
- Future: [`scripts/plot_from_results.py`](../../../scripts/plot_from_results.py) with JSON from benchmarks

Do not cite throughput numbers from these scripts without re-running the
corresponding benchmark harness on your hardware.
