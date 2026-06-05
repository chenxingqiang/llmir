"""Tests for PagedKVCache factory and backend selection."""

import os

import numpy as np
import pytest

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache
from llmir.runtime.kv_factory import create_paged_kv_cache, kv_cache_backend_name


def test_create_paged_kv_cache_numpy_fallback(monkeypatch):
    monkeypatch.setenv("LLMIR_KV_BACKEND", "numpy")
    config = KVCacheConfig(num_layers=2, num_heads=4, head_dim=8, block_size=16)
    cache = create_paged_kv_cache(config, prefer_native=True)
    assert isinstance(cache, PagedKVCache)
    assert kv_cache_backend_name(cache) == "numpy"


def test_numpy_append_lookup_roundtrip():
    config = KVCacheConfig(
        num_layers=1, num_heads=2, head_dim=4, block_size=8, dtype="float32"
    )
    cache = create_paged_kv_cache(config, prefer_native=False)
    keys = np.random.randn(1, 3, 2, 4).astype(np.float32)
    values = np.random.randn(1, 3, 2, 4).astype(np.float32)
    seq_ids = np.array([0], dtype=np.int32)
    block_indices = cache.append(keys, values, seq_ids)
    k_out, v_out = cache.lookup(block_indices, np.array([3], dtype=np.int32))
    np.testing.assert_allclose(k_out[0, :3], keys[0], rtol=1e-5)
    np.testing.assert_allclose(v_out[0, :3], values[0], rtol=1e-5)


@pytest.mark.skipif(
    not os.environ.get("LLMIR_LIB_PATH"),
    reason="native runtime not configured",
)
def test_create_paged_kv_cache_native_when_lib_present():
    from llmir.runtime.native_kvcache import NativePagedKVCache

    config = KVCacheConfig(num_layers=32, num_heads=4, head_dim=8, block_size=16)
    cache = create_paged_kv_cache(config, prefer_native=True)
    assert isinstance(cache, NativePagedKVCache)
    assert kv_cache_backend_name(cache) == "native"
