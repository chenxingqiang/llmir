"""
Factory for PagedKVCache implementations (native C++ or NumPy reference).
"""

from __future__ import annotations

import os
from typing import Optional, Union

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache


def create_paged_kv_cache(
    config: KVCacheConfig,
    *,
    prefer_native: Optional[bool] = None,
) -> PagedKVCache:
    """
  Create a PagedKVCache for one decoder layer.

  Uses the C++ runtime when ``libMLIRLLMRuntime`` is available (see
  ``llmir.runtime.native_bridge``), otherwise the NumPy reference class.

  Set ``LLMIR_KV_BACKEND=numpy`` to force the reference path.
  Set ``LLMIR_LIB_PATH`` to point at the shared library.
  """
    if prefer_native is None:
        backend = os.environ.get("LLMIR_KV_BACKEND", "auto").lower()
        prefer_native = backend in ("auto", "native", "cpp")

    if prefer_native:
        try:
            from llmir.runtime.native_kvcache import NativePagedKVCache

            return NativePagedKVCache(config)
        except RuntimeError:
            pass

    return PagedKVCache(config)


def kv_cache_backend_name(cache: Union[PagedKVCache, object]) -> str:
    """Return a short label for logging/benchmarks."""
    cls = type(cache).__name__
    if cls == "NativePagedKVCache":
        return "native"
    return "numpy"
