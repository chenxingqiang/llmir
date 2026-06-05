"""Store and restore per-layer KV tensors for reusable prompt prefixes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from llmir.runtime.config import KVCacheConfig, PrefixCacheConfig
from llmir.runtime.kv_cache import PagedKVCache
from llmir.runtime.kv_factory import create_paged_kv_cache


@dataclass
class PrefixCacheStats:
    """Counters updated by :class:`PrefixKVStore`."""

    hits: int = 0
    misses: int = 0
    tokens_served_from_cache: int = 0
    entries_stored: int = 0

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class PrefixKVStore:
    """
    Maps token-id prefixes to captured per-layer K/V tensors.

    Used by :class:`PagedKVDecoder` to skip re-running transformer prefill
    for shared prompt prefixes (system prompts, few-shot exemplars, etc.).
    """

    config: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
    _entries: "OrderedDict[Tuple[int, ...], List[Tuple[np.ndarray, np.ndarray]]]" = (
        field(default_factory=OrderedDict, repr=False)
    )

    def clear(self) -> None:
        self._entries.clear()
        self.stats = PrefixCacheStats()

    def store(
        self,
        token_ids: List[int],
        layer_caches: List[PagedKVCache],
        kv_config: KVCacheConfig,
    ) -> bool:
        """Capture KV for ``token_ids`` and all eligible prefixes."""
        if len(token_ids) < self.config.min_prefix_length:
            return False
        stored = False
        for length in range(self.config.min_prefix_length, len(token_ids) + 1):
            key = tuple(int(t) for t in token_ids[:length])
            while len(self._entries) >= self.config.max_prefixes:
                self._entries.popitem(last=False)
            self._entries[key] = _capture_layer_kv(layer_caches, length, kv_config)
            self._entries.move_to_end(key)
            stored = True
        if stored:
            self.stats.entries_stored += 1
        return stored

    def lookup_restore(
        self,
        token_ids: List[int],
        kv_config: KVCacheConfig,
    ) -> Tuple[int, Optional[List[PagedKVCache]]]:
        """
        Find the longest cached prefix of ``token_ids`` and restore layer caches.

        Returns ``(match_length, layer_caches)`` or ``(0, None)`` on miss.
        """
        for length in range(len(token_ids), self.config.min_prefix_length - 1, -1):
            key = tuple(int(t) for t in token_ids[:length])
            payload = self._entries.get(key)
            if payload is None:
                continue
            self._entries.move_to_end(key)
            self.stats.hits += 1
            self.stats.tokens_served_from_cache += length
            return length, _restore_layer_kv(payload, kv_config)
        self.stats.misses += 1
        return 0, None


def _capture_layer_kv(
    layer_caches: List[PagedKVCache],
    seq_len: int,
    kv_config: KVCacheConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    batch_size = 1
    block_indices = np.zeros((batch_size, kv_config.num_layers), dtype=np.int32)
    seq_lens = np.full(batch_size, seq_len, dtype=np.int32)
    captured: List[Tuple[np.ndarray, np.ndarray]] = []
    for layer_cache in layer_caches:
        keys, values = layer_cache.lookup(block_indices, seq_lens)
        captured.append((keys.copy(), values.copy()))
    return captured


def _restore_layer_kv(
    layer_kv: List[Tuple[np.ndarray, np.ndarray]],
    kv_config: KVCacheConfig,
) -> List[PagedKVCache]:
    caches = [
        create_paged_kv_cache(kv_config) for _ in range(kv_config.num_layers)
    ]
    seq_ids = np.array([0], dtype=np.int32)
    for layer_idx, (keys, values) in enumerate(layer_kv):
        if layer_idx < len(caches):
            caches[layer_idx].append(keys, values, seq_ids)
    return caches
