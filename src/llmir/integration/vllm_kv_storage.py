"""Disk-backed KV storage for vLLM KV connector integration."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _token_prefix_key(token_ids: Sequence[int]) -> str:
    payload = np.asarray(list(token_ids), dtype=np.int32).tobytes()
    return hashlib.sha256(payload, usedforsecurity=False).hexdigest()


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    """Align token count down to a multiple of ``block_size`` (vLLM convention)."""
    if num_tokens <= 0 or block_size <= 0:
        return 0
    return (num_tokens - 1) // block_size * block_size


@dataclass
class LLMIRKVStorageConfig:
    """Configuration for :class:`LLMIRKVStorage`."""

    storage_path: str = "/tmp/llmir_vllm_kv"
    min_prefix_length: int = 4


class LLMIRKVStorage:
    """
    Store per-layer K/V tensors keyed by prompt token prefixes.

    Layers are stored as NumPy ``.npz`` files (``kv_cache`` array) under a
    content-addressed directory per prefix, matching vLLM ExampleConnector layout.
    """

    def __init__(self, config: Optional[LLMIRKVStorageConfig] = None):
        self.config = config or LLMIRKVStorageConfig()
        self._root = Path(self.config.storage_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        """Remove stored prefix directories under the storage root."""
        for child in self._root.iterdir():
            if child.is_dir():
                for path in child.glob("*.npz"):
                    path.unlink(missing_ok=True)
                try:
                    child.rmdir()
                except OSError:
                    pass

    def longest_cached_prefix_length(
        self, token_ids: Sequence[int], *, block_size: int = 1
    ) -> int:
        """
        Return the longest cached prefix length (block-aligned when ``block_size`` > 1).
        """
        tokens = [int(t) for t in token_ids]
        if len(tokens) < self.config.min_prefix_length:
            return 0
        limit = len(tokens)
        if block_size > 1:
            limit = align_to_block_size(len(tokens), block_size)
            if limit < self.config.min_prefix_length:
                return 0
        for length in range(limit, self.config.min_prefix_length - 1, -1):
            if self._prefix_has_layers(tuple(tokens[:length])):
                return length
        return 0

    def store_layer_kv(
        self,
        token_ids: Sequence[int],
        layer_name: str,
        kv_cache: np.ndarray,
    ) -> None:
        """Persist extracted KV for ``token_ids`` and eligible sub-prefixes."""
        tokens = [int(t) for t in token_ids]
        seq_len = kv_cache.shape[0] if kv_cache.ndim > 0 else len(tokens)
        for length in range(self.config.min_prefix_length, len(tokens) + 1):
            prefix = tuple(tokens[:length])
            layer_dir = self._prefix_dir(prefix)
            layer_dir.mkdir(parents=True, exist_ok=True)
            path = layer_dir / f"{_safe_layer_filename(layer_name)}.npz"
            end = min(length, seq_len)
            np.savez_compressed(path, kv_cache=kv_cache[:end])

    def load_layer_kv(
        self,
        token_ids: Sequence[int],
        layer_name: str,
    ) -> np.ndarray:
        """Load the ``kv_cache`` array for ``layer_name`` at the given prefix."""
        prefix = tuple(int(t) for t in token_ids)
        path = self._prefix_dir(prefix) / f"{_safe_layer_filename(layer_name)}.npz"
        if not path.exists():
            raise FileNotFoundError(f"No KV layer file at {path}")
        return np.load(path)["kv_cache"]

    def _prefix_has_layers(self, prefix: tuple[int, ...]) -> bool:
        layer_dir = self._prefix_dir(prefix)
        return layer_dir.is_dir() and any(layer_dir.glob("*.npz"))

    def _prefix_dir(self, prefix: tuple[int, ...]) -> Path:
        return self._root / _token_prefix_key(prefix)


def _safe_layer_filename(layer_name: str) -> str:
    return layer_name.replace(os.sep, "_").replace("..", "_")
