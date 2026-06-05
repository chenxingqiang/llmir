"""
PagedKVCache backed by the C++ runtime (same public API as the NumPy class).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache
from llmir.runtime.native_bridge import NativePagedKVCacheHandle


class NativePagedKVCache(PagedKVCache):
    """
    Per-layer KV cache using ``libMLIRLLMRuntime``.

    ``PagedKVDecoder`` allocates one instance per transformer layer; the C++
    handle is created with ``num_layers=1`` internally.
    """

    def __init__(self, config: KVCacheConfig):
        # Do not call PagedKVCache.__init__ (would allocate NumPy structures).
        self.config = config
        self._native = NativePagedKVCacheHandle(
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            block_size=config.block_size,
            max_seq_len=config.max_seq_len,
            dtype=config.dtype,
            enable_gpu=config.enable_gpu,
        )
        self._last_block_indices: Optional[np.ndarray] = None

    def append(
        self, keys: np.ndarray, values: np.ndarray, seq_ids: np.ndarray
    ) -> np.ndarray:
        block_indices = self._native.append(keys, values, seq_ids)
        # Match NumPy API: [batch, num_layers] with seq_id in column 0.
        out = np.zeros((keys.shape[0], self.config.num_layers), dtype=np.int32)
        out[:, 0] = seq_ids.astype(np.int32)
        self._last_block_indices = block_indices
        return out

    def lookup(
        self, block_indices: np.ndarray, seq_lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._last_block_indices is not None:
            return self._native.lookup(self._last_block_indices, seq_lens)
        return self._native.lookup(block_indices, seq_lens)

    def clear_sequence(self, seq_id: int) -> bool:
        return self._native.clear_sequence(seq_id)

    def reset(self) -> None:
        self._native.reset()
        self._last_block_indices = None

    def get_memory_usage(self) -> int:
        return self._native.get_memory_usage()

    def get_num_sequences(self) -> int:
        return self._native.get_num_sequences()

    def get_sequence_length(self, seq_id: int) -> int:
        return self._native.get_sequence_length(seq_id)

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def num_layers(self) -> int:
        return self.config.num_layers
