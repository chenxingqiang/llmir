"""MVP-C: torch-GPU KV path through PagedKVDecoder (no CPU NumPy round-trip)."""

from __future__ import annotations

import os

import pytest

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_factory import create_paged_kv_cache, kv_cache_backend_name


def test_kv_factory_torch_cuda_backend(kv_config_small):
    pytest.importorskip("torch")
    os.environ["LLMIR_KV_BACKEND"] = "torch_cuda"
    try:
        cache = create_paged_kv_cache(kv_config_small, device="cpu")
        assert kv_cache_backend_name(cache) == "torch_cuda"
    finally:
        os.environ.pop("LLMIR_KV_BACKEND", None)


def test_paged_decoder_torch_kv_path(tiny_llama, monkeypatch):
    """Decode with torch_cuda backend stays on torch tensors in append/lookup."""
    pytest.importorskip("torch")
    import torch

    model, tokenizer, config = tiny_llama
    os.environ["LLMIR_KV_BACKEND"] = "torch_cuda"
    try:
        from llmir.runtime.paged_decoder import PagedKVDecoder
        from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

        append_types: list = []

        real_append = TorchGpuPagedKVCache.append

        def spy_append(self, keys, values, seq_ids):
            append_types.append((type(keys).__name__, type(values).__name__))
            return real_append(self, keys, values, seq_ids)

        monkeypatch.setattr(TorchGpuPagedKVCache, "append", spy_append)

        decoder = PagedKVDecoder(model, tokenizer)
        caches = decoder._create_layer_caches()
        assert all(isinstance(c, TorchGpuPagedKVCache) for c in caches)

        results = decoder.decode(["hello"], max_new_tokens=2)
        assert len(results) == 1
        assert len(results[0].generated_token_ids) >= 1
        assert append_types
        assert all(t == "Tensor" for pair in append_types for t in pair)
    finally:
        os.environ.pop("LLMIR_KV_BACKEND", None)


def test_paged_decoder_chains_past_key_values_without_repeated_lookup(
    tiny_llama, monkeypatch
):
    """Decode reuses HF past_key_values instead of rebuilding DynamicCache each step."""
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    os.environ["LLMIR_KV_BACKEND"] = "torch_cuda"
    try:
        from llmir.runtime.paged_decoder import PagedKVDecoder

        lookup_calls = {"n": 0}
        real_lookup = PagedKVDecoder._lookup_dynamic_cache

        def counted_lookup(self, layer_caches, past_len):
            lookup_calls["n"] += 1
            return real_lookup(self, layer_caches, past_len)

        monkeypatch.setattr(
            PagedKVDecoder, "_lookup_dynamic_cache", counted_lookup
        )

        decoder = PagedKVDecoder(model, tokenizer, enable_prefix_cache=False)
        decoder.decode(["hello world"], max_new_tokens=4)
        # At most one lookup (prefill start); decode steps must chain past_key_values.
        assert lookup_calls["n"] <= 1
    finally:
        os.environ.pop("LLMIR_KV_BACKEND", None)


def test_cuda_probe_summarize():
    from llmir.runtime.cuda_probe import summarize_cuda_stack

    stack = summarize_cuda_stack()
    assert "torch_cuda" in stack
    assert "native_cuda_built" in stack
    assert "device_count" in stack


@pytest.fixture
def kv_config_small():
    return KVCacheConfig(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        block_size=4,
        max_seq_len=32,
        dtype="float32",
        enable_gpu=False,
    )
