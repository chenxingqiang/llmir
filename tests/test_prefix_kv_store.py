"""Tests for PrefixKVStore capture/restore."""

import numpy as np

from llmir.runtime.config import KVCacheConfig, PrefixCacheConfig
from llmir.runtime.kv_factory import create_paged_kv_cache
from llmir.runtime.prefix_kv_store import PrefixKVStore


def test_store_and_restore_roundtrip():
    config = KVCacheConfig(
        num_layers=2, num_heads=2, head_dim=4, block_size=8, dtype="float32"
    )
    store = PrefixKVStore(PrefixCacheConfig(min_prefix_length=2))
    caches = [create_paged_kv_cache(config) for _ in range(2)]
    keys = np.random.randn(1, 5, 2, 4).astype(np.float32)
    values = np.random.randn(1, 5, 2, 4).astype(np.float32)
    seq_ids = np.array([0], dtype=np.int32)
    for cache in caches:
        cache.append(keys, values, seq_ids)

    token_ids = [10, 11, 12, 13, 14]
    assert store.store(token_ids, caches, config)

    match_len, restored = store.lookup_restore(token_ids[:4], config)
    assert match_len == 4
    assert restored is not None
    assert restored[0].get_sequence_length(0) == 4

    match_len, restored = store.lookup_restore(token_ids, config)
    assert match_len == 5
    assert restored is not None
    assert len(restored) == 2
    assert restored[0].get_sequence_length(0) == 5
