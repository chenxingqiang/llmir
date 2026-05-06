# `cpu_inference_compare.py` — CPU comparison benchmark

Four-way comparison on CPU between:

1. `llmir` — LLMIR's built-in scheduler with the placeholder token loop. No
   real model is executed; this row measures Python serving overhead only and
   is **not** comparable to the others.
2. `vllm` — vLLM's `LLM` engine running directly. vLLM owns the KV cache and
   attention kernels; this is the upstream baseline.
3. `llmir+vllm` — LLMIR's `LLMEngine` with `backend="vllm"`. This path
   currently **forwards** to `vllm.LLM.generate()`. LLMIR is *not* in the hot
   loop, so by construction this row mirrors the `vllm` row to within Python
   wrapper overhead. Useful to check that the bridge is regression-free; it
   does **not** reflect any LLMIR optimization.
4. `llmir-paged` — **Kernel-layer integration.** LLMIR's `LLMEngine` with
   `backend="llmir_paged"`. Drives a HuggingFace `transformers` model in a
   manual decode loop where every layer's K/V tensors flow through
   `llmir.runtime.PagedKVCache` between forward steps. This is the row that
   actually exercises LLMIR's KV-cache subsystem and where future LLMIR
   optimizations (paged storage, quantization, prefix sharing, speculative
   branches) will surface as deltas vs. the `vllm` baseline.

## What each row measures (and what it does *not*)

| Row | Model executor | KV cache subsystem | Useful for |
|---|---|---|---|
| `llmir` | none (placeholder) | none | sanity-checking serving plumbing |
| `vllm` | vLLM | vLLM | upstream baseline |
| `llmir+vllm` | vLLM | vLLM | regression check on the bridge layer |
| `llmir-paged` | HF transformers | **LLMIR `PagedKVCache`** | measuring LLMIR KV-cache impact |

The reference numbers further down show `vllm` ≈ `llmir+vllm` to within ~1 %,
which is *expected*: when LLMIR isn't in the hot loop, there is nothing for
it to optimize. Use the `llmir-paged` row to evaluate LLMIR's actual impact.

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

The `llmir-paged` row needs `transformers` and `torch` (and is independent of
vLLM). Install those if you want to run only the kernel-integrated row:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers torch
```

## Run

```bash
PYTHONPATH=src python scripts/cpu_inference_compare.py \
    --model facebook/opt-125m \
    --batch-size 4 --prompt-tokens 32 --max-tokens 32 \
    --warmup 1 --output bench.json

# Skip rows that require missing dependencies:
PYTHONPATH=src python scripts/cpu_inference_compare.py \
    --skip-vllm --skip-llmir-vllm-backend          # only llmir + llmir-paged
PYTHONPATH=src python scripts/cpu_inference_compare.py \
    --skip-llmir-paged                              # legacy three-way
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

These are the legacy three-row results; `llmir+vllm` ≈ `vllm` because the bridge
forwards every call to vLLM. The `llmir-paged` row was added to expose LLMIR's
KV-cache subsystem on the critical path; capture and report its numbers
separately when comparing kernel-level optimizations.

The `llmir` row in the JSON file reports the LLMIR serving path's own overhead
without invoking a real LLM and is **not** comparable to the vLLM rows.
