# P5: CUDA kernels and native runtime packaging

## CUDA kernels (`CUDAKernels.cu`)

When `LLMIR_ENABLE_CUDA=ON`, `libMLIRLLMRuntime` links:

- **Multi-query attention** (`launchMultiQueryAttentionKernel`) — block-wise softmax attention matching the CPU MQA layout
- **Grouped-query / pruned / flash** launchers route through the same kernel (MVP)

Build:

```bash
export LLMIR_ENABLE_CUDA=ON   # or auto-detect nvcc
./scripts/build_native_runtime.sh
export LLMIR_LIB_PATH=build-native/.../libMLIRLLMRuntime.so
```

## Python probes

```python
from llmir.runtime.cuda_probe import summarize_cuda_stack

print(summarize_cuda_stack())
# torch_cuda, native_cuda_built, native_cuda_runtime, device_count
```

C API:

- `llmir_has_cuda_support()` — compiled with CUDA
- `llmir_cuda_runtime_available()` — CUDA build + visible device
- `llmir_cuda_device_count()`

## Package native library into the wheel tree

```bash
./scripts/build_native_runtime.sh
./scripts/package_native_lib.sh "$LLMIR_LIB_PATH"
pip install -e ".[native]"
```

Prebuilt PyPI wheels shipping `libMLIRLLMRuntime.so` per platform remain **planned**; this script supports local/dev wheels.

## Limitations (MVP)

- MQA kernel uses one thread per (batch, seq, head); not yet FlashAttention-tiled
- No automatic GPU KV block copy into vLLM pools (see P4 connector)
- Full LLVM+MLIR tree build is heavy; CI keeps `continue-on-error`
