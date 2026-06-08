"""Store and restore per-layer KV tensors for reusable prompt prefixes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from llmir.runtime.config import KVCacheConfig, PrefixCacheConfig
from llmir.runtime.kv_cache import PagedKVCache
from llmir.runtime.kv_factory import create_paged_kv_cache

# Stored K/V payload: NumPy (CPU) or torch.Tensor (GPU-resident).
LayerKVPair = Tuple[Any, Any]


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
    _entries: "OrderedDict[Tuple[int, ...], List[LayerKVPair]]" = field(
        default_factory=OrderedDict, repr=False
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
        *,
        device: Optional[str] = None,
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
            return length, _restore_layer_kv(payload, kv_config, device=device)
        self.stats.misses += 1
        return 0, None


def _is_torch_tensor(data: object) -> bool:
    try:
        import torch

        return isinstance(data, torch.Tensor)
    except ImportError:
        return False


def _capture_layer_kv(
    layer_caches: List[PagedKVCache],
    seq_len: int,
    kv_config: KVCacheConfig,
) -> List[LayerKVPair]:
    from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

    batch_size = 1
    block_indices = np.zeros((batch_size, kv_config.num_layers), dtype=np.int32)
    seq_lens = np.full(batch_size, seq_len, dtype=np.int32)
    captured: List[LayerKVPair] = []
    for layer_cache in layer_caches:
        if isinstance(layer_cache, TorchGpuPagedKVCache):
            keys, values = layer_cache.export_dense(0, seq_len)
            captured.append((keys.clone(), values.clone()))
            continue
        keys, values = layer_cache.lookup(block_indices, seq_lens)
        captured.append((_to_numpy(keys), _to_numpy(values)))
    return captured


def _restore_layer_kv(
    layer_kv: List[LayerKVPair],
    kv_config: KVCacheConfig,
    *,
    device: Optional[str] = None,
) -> List[PagedKVCache]:
    from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

    caches = [
        create_paged_kv_cache(kv_config, device=device)
        for _ in range(kv_config.num_layers)
    ]
    seq_ids = np.array([0], dtype=np.int32)
    for layer_idx, (keys, values) in enumerate(layer_kv):
        if layer_idx >= len(caches):
            break
        cache = caches[layer_idx]
        if isinstance(cache, TorchGpuPagedKVCache) and _is_torch_tensor(keys):
            cache.import_dense(keys, values, 0)
        else:
            cache.append(keys, values, seq_ids)
    return caches


def _to_numpy(data: object) -> np.ndarray:
    """Normalize torch or NumPy KV tensors for CPU prefix persistence."""
    if _is_torch_tensor(data):
        return data.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(data).copy()
