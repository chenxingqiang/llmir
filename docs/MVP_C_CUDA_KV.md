# MVP-C: Native CUDA KV (GPU-resident path)

Removes the **CPU NumPy round-trip** in `PagedKVDecoder` when the model runs on CUDA. This is the main latency fix for `llmir-paged` on GPU; paper-scale wins still come from prefix reuse (MVP-B) and compile-time block sizing (MVP-A).

## Backends (`LLMIR_KV_BACKEND`)

| Value | Implementation | When |
|-------|----------------|------|
| `numpy` | `PagedKVCache` (CPU NumPy) | Reference / CPU |
| `torch_cuda` | `TorchGpuPagedKVCache` | GPU tensors, no host copy |
| `native` | `libMLIRLLMRuntime` + CUDA kernels | `LLMIR_LIB_PATH` + `LLMIR_ENABLE_CUDA` build |
| `auto` | native → torch_cuda (if CUDA) → numpy | Default |

## Commands

```bash
# Unit tests (no GPU required)
pytest tests/test_torch_gpu_kv_cache.py tests/test_mvp_c_e2e.py -m "not network" -q

# Compare numpy vs torch_cuda on CPU (functional) or GPU (performance)
python scripts/mvp_c_cuda_kv_bench.py --model gpt2 --device cuda -o mvp_c.json

llmir-benchmark --mvp-c-bench --model gpt2 -o mvp_c.json

# Native CUDA runtime (optional)
LLMIR_ENABLE_CUDA=ON ./scripts/build_native_runtime.sh
export LLMIR_LIB_PATH=$PWD/build/lib/libMLIRLLMRuntime.so
LLMIR_KV_BACKEND=native python scripts/mvp_c_cuda_kv_bench.py --backends numpy,native
```

## CUDA probes

```python
from llmir.runtime.cuda_probe import summarize_cuda_stack
print(summarize_cuda_stack())
# torch_cuda, native_cuda_built, native_cuda_runtime, device_count
```

## Architecture

```mermaid
flowchart LR
  HF[HF forward K/V] -->|GPU tensors| Append[PagedKVDecoder._append_to_paged]
  Append -->|torch_cuda| TKV[TorchGpuPagedKVCache]
  Append -->|numpy| NKV[PagedKVCache CPU]
  TKV --> Lookup[_lookup_dynamic_cache]
  Lookup --> DCache[transformers DynamicCache]
```

## Success criteria

- **Correctness**: `test_mvp_c_e2e` — decode with `LLMIR_KV_BACKEND=torch_cuda` passes; append receives `torch.Tensor`, not NumPy.
- **Performance (GPU)**: `torch_cuda` tok/s ≥ `numpy` on the same prompt (MVP-C bench); larger gap on longer prompts / more layers.
- **Optional**: `native` backend matches or beats `torch_cuda` when `libMLIRLLMRuntime` is built with CUDA.

## Honesty

- `TorchGpuPagedKVCache` is a **concat-list** store (MVP), not full block-paged GPU memory from the C++ runtime.
- Prefix restore still serializes to NumPy in `PrefixKVStore`; GPU path re-uploads on restore (acceptable for MVP-C).
- Paper Table II multi-model vLLM numbers are still out of scope.
