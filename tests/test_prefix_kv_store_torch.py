"""PrefixKVStore with GPU-resident torch KV (no NumPy round-trip)."""

from __future__ import annotations

import os

import pytest

from llmir.runtime.config import KVCacheConfig, PrefixCacheConfig
from llmir.runtime.prefix_kv_store import PrefixKVStore
from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache


@pytest.fixture
def kv_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=2,
        num_heads=2,
        head_dim=4,
        block_size=4,
        dtype="float32",
        enable_gpu=False,
    )


def test_prefix_store_torch_roundtrip_stays_on_device(kv_config):
    pytest.importorskip("torch")
    import torch

    os.environ["LLMIR_KV_BACKEND"] = "torch_cuda"
    try:
        store = PrefixKVStore(PrefixCacheConfig(min_prefix_length=2))
        caches = [
            TorchGpuPagedKVCache(kv_config, device="cpu") for _ in range(2)
        ]
        keys = torch.randn(1, 5, kv_config.num_heads, kv_config.head_dim)
        values = torch.randn(1, 5, kv_config.num_heads, kv_config.head_dim)
        for cache in caches:
            cache.import_dense(keys, values, 0)

        token_ids = [10, 11, 12, 13, 14]
        assert store.store(token_ids, caches, kv_config)

        payload = store._entries[tuple(token_ids[:4])]
        assert all(isinstance(pair[0], torch.Tensor) for pair in payload)
        assert payload[0][0].device.type == "cpu"

        match_len, restored = store.lookup_restore(
            token_ids[:4], kv_config, device="cpu"
        )
        assert match_len == 4
        assert restored is not None
        assert all(isinstance(c, TorchGpuPagedKVCache) for c in restored)
        ek, ev = restored[0].export_dense(0, 4)
        assert torch.allclose(ek[0, :4], keys[0, :4])
        assert torch.allclose(ev[0, :4], values[0, :4])
    finally:
        os.environ.pop("LLMIR_KV_BACKEND", None)
