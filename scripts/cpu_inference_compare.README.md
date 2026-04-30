# `cpu_inference_compare.py` — CPU comparison benchmark

Three-way comparison on CPU between:

1. `llmir` — LLMIR's serving/token path (no real model; placeholder)
2. `vllm` — vLLM's `LLM` engine running directly
3. `llmir+vllm` — LLMIR's `LLMEngine` driving the optional vLLM backend

## Install vLLM on CPU

vLLM's PyPI default wheel is the CUDA build and will not run on a GPU-less host.
Use the official **pre-built CPU wheel** from the GitHub release (vLLM ≥ 0.17.0
publishes them) or build from source with `VLLM_TARGET_DEVICE=cpu`.

### Pre-built CPU wheel (recommended)

```bash
# 1) Install torch (CPU-only build is preferred but the default PyPI wheel works
#    on CPU when no GPU is present; vLLM imports torch's CPU APIs only).
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.11.0+cpu" torchaudio torchvision

# 2) Download and install the CPU vLLM wheel for the matching version.
VLLM_VERSION=0.20.0
pip install \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl"

# 3) (Optional) Add TCMalloc + Intel OpenMP to LD_PRELOAD for best CPU perf.
sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4
TC=$(find / -iname 'libtcmalloc_minimal.so.4' 2>/dev/null | head -1)
IOMP=$(find / -iname 'libiomp5.so' 2>/dev/null | head -1)
export LD_PRELOAD="${TC}:${IOMP}:${LD_PRELOAD}"
```

### Build from source

```bash
git clone --branch v0.20.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/cpu.txt       --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu pip install -v --no-build-isolation .
```

## Run

```bash
PYTHONPATH=src python scripts/cpu_inference_compare.py \
    --model facebook/opt-125m \
    --batch-size 4 --prompt-tokens 32 --max-tokens 32 \
    --warmup 1 --output bench.json
```

When the host has no GPU, vLLM auto-selects `CpuPlatform`. `device_type='cpu'`
is set automatically and **must not** be passed as a kwarg to `LLM(...)` (it was
removed from `EngineArgs` in vLLM 0.20).

## Reference numbers (CPU, no AVX-512)

The numbers in [`cpu_inference_compare.results.json`](./cpu_inference_compare.results.json)
were captured on a 4-core x86_64 sandbox (AVX2, no AVX-512, 15 GiB RAM, no GPU)
running the script against a randomly initialized 2-layer / hidden=128 OPT model
exported to `/tmp/tiny-opt`:

| batch | prompt | gen | `vllm` Tok/s | `llmir+vllm` Tok/s | overhead |
|------:|-------:|----:|-------------:|-------------------:|---------:|
|     1 |     16 |  16 |       212.26 |             211.38 |   +0.4%  |
|     4 |     32 | 128 |       616.63 |             612.74 |   +0.6%  |
|     8 |     64 | 512 |     1,045.87 |           1,085.63 |   −3.7%  |

The `llmir` row in the JSON file reports the LLMIR serving path's own overhead
without invoking a real LLM and is **not** comparable to the vLLM rows.
