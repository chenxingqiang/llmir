"""
Factory for PagedKVCache implementations (native C++, torch GPU, or NumPy reference).
"""

from __future__ import annotations

import os
from typing import Optional, Union

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache


def _env_backend() -> str:
    return os.environ.get("LLMIR_KV_BACKEND", "auto").lower()


def create_paged_kv_cache(
    config: KVCacheConfig,
    *,
    prefer_native: Optional[bool] = None,
    device: Optional[str] = None,
) -> PagedKVCache:
    """
    Create a PagedKVCache for one decoder layer.

    Backend selection (``LLMIR_KV_BACKEND``):

    - ``numpy`` — reference NumPy store (CPU copies from decoder)
    - ``torch`` / ``torch_cuda`` — GPU-resident torch tensors (MVP-C default on CUDA)
    - ``native`` / ``cpp`` — ``libMLIRLLMRuntime`` when ``LLMIR_LIB_PATH`` is set
    - ``auto`` — native if library loads, else torch GPU when CUDA visible, else numpy
    """
    backend = _env_backend()

    if backend == "numpy":
        return PagedKVCache(config)

    if backend in ("torch", "torch_cuda", "cuda"):
        from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

        return TorchGpuPagedKVCache(config, device=device)  # type: ignore[return-value]

    use_native = prefer_native
    if use_native is None:
        use_native = backend in ("auto", "native", "cpp")

    if use_native and backend != "torch":
        try:
            from llmir.runtime.native_kvcache import NativePagedKVCache

            return NativePagedKVCache(config)
        except RuntimeError:
            pass

    if backend in ("auto", "native", "cpp") and _should_use_torch_gpu(config, device):
        from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

        return TorchGpuPagedKVCache(config, device=device)  # type: ignore[return-value]

    return PagedKVCache(config)


def _should_use_torch_gpu(config: KVCacheConfig, device: Optional[str]) -> bool:
    if device == "cpu":
        return False
    if device == "cuda":
        return True
    if not config.enable_gpu:
        return False
    try:
        from llmir.runtime.torch_gpu_kv_cache import torch_cuda_available

        return torch_cuda_available()
    except ImportError:
        return False


def kv_cache_backend_name(cache: Union[PagedKVCache, object]) -> str:
    """Return a short label for logging/benchmarks."""
    cls = type(cache).__name__
    if cls == "NativePagedKVCache":
        return "native"
    if cls == "TorchGpuPagedKVCache":
        return "torch_cuda"
    return "numpy"
