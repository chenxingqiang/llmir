"""Tests for TorchGpuPagedKVCache (MVP-C)."""

from __future__ import annotations

import numpy as np
import pytest

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.torch_gpu_kv_cache import (
    TorchGpuPagedKVCache,
    hf_kv_to_llmir_layout,
    llmir_kv_to_hf_layout,
)


@pytest.fixture
def kv_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=1,
        num_heads=2,
        head_dim=8,
        block_size=4,
        max_seq_len=32,
        dtype="float32",
        enable_gpu=False,
    )


def test_layout_round_trip():
    pytest.importorskip("torch")
    import torch

    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    k_ll, v_ll = hf_kv_to_llmir_layout(k, v)
    assert k_ll.shape == (1, 5, 2, 8)
    k_hf, v_hf = llmir_kv_to_hf_layout(k_ll, v_ll)
    assert torch.allclose(k, k_hf)
    assert torch.allclose(v, v_hf)


def test_torch_gpu_cache_append_lookup_numpy(kv_config):
    pytest.importorskip("torch")
    cache = TorchGpuPagedKVCache(kv_config, device="cpu")
    keys = np.random.randn(1, 3, kv_config.num_heads, kv_config.head_dim).astype(
        np.float32
    )
    values = np.random.randn(1, 3, kv_config.num_heads, kv_config.head_dim).astype(
        np.float32
    )
    seq_ids = np.array([0], dtype=np.int32)
    cache.append(keys, values, seq_ids)
    block_indices = np.zeros((1, kv_config.num_layers), dtype=np.int32)
    seq_lens = np.array([3], dtype=np.int32)
    k_out, v_out = cache.lookup(block_indices, seq_lens)
    assert k_out.shape == (1, 3, kv_config.num_heads, kv_config.head_dim)
    assert np.allclose(k_out.cpu().numpy(), keys)
    assert np.allclose(v_out.cpu().numpy(), values)


def test_torch_gpu_cache_append_lookup_torch(kv_config):
    pytest.importorskip("torch")
    import torch

    cache = TorchGpuPagedKVCache(kv_config, device="cpu")
    keys = torch.randn(1, 2, kv_config.num_heads, kv_config.head_dim)
    values = torch.randn(1, 2, kv_config.num_heads, kv_config.head_dim)
    cache.append(keys, values, np.array([0], dtype=np.int32))
    k_out, v_out = cache.lookup(
        np.zeros((1, 1), dtype=np.int32), np.array([2], dtype=np.int32)
    )
    assert torch.allclose(k_out[0, :2], keys[0])
    assert torch.allclose(v_out[0, :2], values[0])
