"""
GPU-resident block-paged PagedKVCache using PyTorch tensors (MVP-C).

Mirrors the block allocator layout of ``lib/Dialect/LLM/Runtime/KVCache.cpp``:
fixed-size token blocks per layer, with free-list reuse. Avoids concat-list
storage and CPU NumPy round-trips on the decode hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from llmir.runtime.config import KVCacheConfig

try:
    import torch
except ImportError:  # pragma: no cover - optional
    torch = None  # type: ignore[assignment]


def torch_cuda_available() -> bool:
    if torch is None:
        return False
    return bool(torch.cuda.is_available())


def hf_kv_to_llmir_layout(
    key: "torch.Tensor", value: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """(batch, heads, seq, dim) -> (batch, seq, heads, dim)."""
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


def llmir_kv_to_hf_layout(
    key: "torch.Tensor", value: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """(batch, seq, heads, dim) -> (batch, heads, seq, dim)."""
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


@dataclass
class _KVBlock:
    """One fixed-size K/V block (``block_size`` tokens)."""

    keys: "torch.Tensor"
    values: "torch.Tensor"
    used_slots: int = 0


@dataclass
class _SequenceState:
    """Per-sequence block table (ordered block ids + tail position)."""

    length: int = 0
    block_ids: List[int] = field(default_factory=list)
    last_block_id: int = -1
    pos_in_last_block: int = 0


class TorchGpuPagedKVCache:
    """
    Per-layer block-paged KV store backed by GPU (or CPU) torch tensors.

    Public API mirrors :class:`llmir.runtime.kv_cache.PagedKVCache`.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        *,
        device: Optional[Union[str, "torch.device"]] = None,
    ):
        if torch is None:
            raise RuntimeError("TorchGpuPagedKVCache requires PyTorch")
        self.config = config
        if device is None:
            device = torch.device(
                "cuda" if config.enable_gpu and torch_cuda_available() else "cpu"
            )
        self.device = torch.device(device)
        self._dtype = self._resolve_dtype(config.dtype)
        self._blocks: List[_KVBlock] = []
        self._free_block_ids: List[int] = []
        self._sequences: Dict[int, _SequenceState] = {}

    @staticmethod
    def _resolve_dtype(dtype: str) -> "torch.dtype":
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype.lower(), torch.float32)

    def append(
        self,
        keys: Union["torch.Tensor", np.ndarray],
        values: Union["torch.Tensor", np.ndarray],
        seq_ids: Union[np.ndarray, "torch.Tensor"],
    ) -> np.ndarray:
        keys_t = self._as_tensor(keys)
        values_t = self._as_tensor(values)
        batch_size = keys_t.shape[0]
        seq_len = keys_t.shape[1]
        if keys_t.shape != values_t.shape:
            raise ValueError("keys and values must have the same shape")

        if isinstance(seq_ids, torch.Tensor):
            seq_list = [int(x) for x in seq_ids.detach().cpu().tolist()]
        else:
            seq_list = [int(x) for x in seq_ids]

        block_indices = np.zeros((batch_size, self.config.num_layers), dtype=np.int32)
        for batch_idx, seq_id in enumerate(seq_list):
            block_indices[batch_idx, 0] = seq_id
            self._append_tokens(seq_id, keys_t[batch_idx], values_t[batch_idx])
        return block_indices

    def lookup(
        self,
        block_indices: np.ndarray,
        seq_lens: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        batch_size = len(seq_lens)
        max_seq_len = int(seq_lens.max()) if len(seq_lens) else 0
        keys = torch.zeros(
            (batch_size, max_seq_len, self.config.num_heads, self.config.head_dim),
            dtype=self._dtype,
            device=self.device,
        )
        values = torch.zeros_like(keys)

        for batch_idx, seq_len in enumerate(seq_lens):
            seq_id = int(block_indices[batch_idx, 0])
            self._read_dense_into(seq_id, int(seq_len), keys[batch_idx], values[batch_idx])
        return keys, values

    def export_dense(
        self, seq_id: int, seq_len: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Return ``(1, seq_len, heads, dim)`` tensors on ``self.device``."""
        keys = torch.zeros(
            (1, seq_len, self.config.num_heads, self.config.head_dim),
            dtype=self._dtype,
            device=self.device,
        )
        values = torch.zeros_like(keys)
        self._read_dense_into(seq_id, seq_len, keys[0], values[0])
        return keys, values

    def import_dense(
        self,
        keys: Union["torch.Tensor", np.ndarray],
        values: Union["torch.Tensor", np.ndarray],
        seq_id: int,
    ) -> None:
        """Load dense ``(batch, seq, heads, dim)`` into block storage for ``seq_id``."""
        keys_t = self._as_tensor(keys)
        values_t = self._as_tensor(values)
        if keys_t.shape[0] != 1:
            raise ValueError("import_dense supports batch size 1")
        self.clear_sequence(seq_id)
        self._append_tokens(seq_id, keys_t[0], values_t[0])

    def clear_sequence(self, seq_id: int) -> bool:
        state = self._sequences.pop(seq_id, None)
        if state is None:
            return False
        for block_id in state.block_ids:
            if 0 <= block_id < len(self._blocks):
                self._blocks[block_id].used_slots = 0
                self._free_block_ids.append(block_id)
        return True

    def reset(self) -> None:
        self._sequences.clear()
        self._blocks.clear()
        self._free_block_ids.clear()

    def get_memory_usage(self) -> int:
        total = 0
        for block in self._blocks:
            total += int(block.keys.element_size() * block.keys.nelement())
            total += int(block.values.element_size() * block.values.nelement())
        return total

    def get_num_sequences(self) -> int:
        return len(self._sequences)

    def get_sequence_length(self, seq_id: int) -> int:
        state = self._sequences.get(seq_id)
        return state.length if state is not None else 0

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def _alloc_block(self) -> int:
        if self._free_block_ids:
            block_id = self._free_block_ids.pop()
            self._blocks[block_id].used_slots = 0
            return block_id
        block = _KVBlock(
            keys=torch.zeros(
                self.config.block_size,
                self.config.num_heads,
                self.config.head_dim,
                dtype=self._dtype,
                device=self.device,
            ),
            values=torch.zeros(
                self.config.block_size,
                self.config.num_heads,
                self.config.head_dim,
                dtype=self._dtype,
                device=self.device,
            ),
        )
        self._blocks.append(block)
        return len(self._blocks) - 1

    def _sequence_state(self, seq_id: int) -> _SequenceState:
        if seq_id not in self._sequences:
            self._sequences[seq_id] = _SequenceState()
        return self._sequences[seq_id]

    def _append_tokens(
        self, seq_id: int, keys: "torch.Tensor", values: "torch.Tensor"
    ) -> None:
        """Append ``(seq_len, heads, dim)`` slices into block storage."""
        state = self._sequence_state(seq_id)
        seq_len = int(keys.shape[0])
        offset = 0
        while offset < seq_len:
            if (
                state.last_block_id < 0
                or state.pos_in_last_block >= self.config.block_size
            ):
                block_id = self._alloc_block()
                state.block_ids.append(block_id)
                state.last_block_id = block_id
                state.pos_in_last_block = 0

            block = self._blocks[state.last_block_id]
            space = self.config.block_size - state.pos_in_last_block
            take = min(seq_len - offset, space)
            start = state.pos_in_last_block
            end = start + take
            block.keys[start:end] = keys[offset : offset + take]
            block.values[start:end] = values[offset : offset + take]
            block.used_slots = max(block.used_slots, end)
            state.pos_in_last_block = end
            state.length += take
            offset += take

    def _read_dense_into(
        self,
        seq_id: int,
        seq_len: int,
        keys_out: "torch.Tensor",
        values_out: "torch.Tensor",
    ) -> None:
        if seq_len <= 0:
            return
        state = self._sequences.get(seq_id)
        if state is None or not state.block_ids:
            return
        written = 0
        for block_id in state.block_ids:
            if written >= seq_len:
                break
            block = self._blocks[block_id]
            take = min(seq_len - written, block.used_slots)
            if take <= 0:
                continue
            keys_out[written : written + take] = block.keys[:take]
            values_out[written : written + take] = block.values[:take]
            written += take

    def _as_tensor(self, data: Union["torch.Tensor", np.ndarray]) -> "torch.Tensor":
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self._dtype)
        return torch.from_numpy(np.asarray(data)).to(device=self.device, dtype=self._dtype)
