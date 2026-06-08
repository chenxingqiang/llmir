"""
GPU-resident PagedKVCache using PyTorch tensors (MVP-C).

Avoids the CPU NumPy round-trip in :mod:`llmir.runtime.paged_decoder` when the
model runs on CUDA. Native ``libMLIRLLMRuntime`` GPU blocks are used when
``LLMIR_KV_BACKEND=native`` and CUDA runtime probes succeed; otherwise this
module is the default CUDA path.
"""

from __future__ import annotations

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


def hf_kv_to_llmir_layout(key: "torch.Tensor", value: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """(batch, heads, seq, dim) -> (batch, seq, heads, dim)."""
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


def llmir_kv_to_hf_layout(key: "torch.Tensor", value: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """(batch, seq, heads, dim) -> (batch, heads, seq, dim)."""
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


class TorchGpuPagedKVCache:
    """
    Per-layer KV store backed by GPU (or CPU) torch tensors.

    Public API mirrors :class:`llmir.runtime.kv_cache.PagedKVCache`.
    """

    def __init__(self, config: KVCacheConfig, *, device: Optional[Union[str, "torch.device"]] = None):
        if torch is None:
            raise RuntimeError("TorchGpuPagedKVCache requires PyTorch")
        self.config = config
        if device is None:
            device = torch.device("cuda" if config.enable_gpu and torch_cuda_available() else "cpu")
        self.device = torch.device(device)
        self._dtype = self._resolve_dtype(config.dtype)
        self._sequences: Dict[int, Dict[str, object]] = {}

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
        for i, seq_id in enumerate(seq_list):
            block_indices[i, 0] = seq_id
            if seq_id not in self._sequences:
                self._sequences[seq_id] = {"length": 0, "kv_list": []}
            entry = self._sequences[seq_id]
            entry["length"] = int(entry["length"]) + seq_len
            kv_list = entry["kv_list"]
            assert isinstance(kv_list, list)
            kv_list.append((keys_t[i : i + 1].clone(), values_t[i : i + 1].clone()))

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

        for i, seq_len in enumerate(seq_lens):
            seq_id = int(block_indices[i, 0])
            seq_len = int(seq_len)
            entry = self._sequences.get(seq_id)
            if entry is None:
                continue
            kv_list = entry["kv_list"]
            assert isinstance(kv_list, list)
            if not kv_list:
                continue
            k_cat = torch.cat([kv[0] for kv in kv_list], dim=1)
            v_cat = torch.cat([kv[1] for kv in kv_list], dim=1)
            take = min(seq_len, k_cat.shape[1])
            keys[i, :take] = k_cat[0, :take]
            values[i, :take] = v_cat[0, :take]

        return keys, values

    def clear_sequence(self, seq_id: int) -> bool:
        if seq_id in self._sequences:
            del self._sequences[seq_id]
            return True
        return False

    def reset(self) -> None:
        self._sequences.clear()

    def get_memory_usage(self) -> int:
        total = 0
        for entry in self._sequences.values():
            kv_list = entry["kv_list"]
            assert isinstance(kv_list, list)
            for k, v in kv_list:
                total += int(k.element_size() * k.nelement())
                total += int(v.element_size() * v.nelement())
        return total

    def get_num_sequences(self) -> int:
        return len(self._sequences)

    def get_sequence_length(self, seq_id: int) -> int:
        entry = self._sequences.get(seq_id)
        if entry is None:
            return 0
        return int(entry["length"])

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def _as_tensor(self, data: Union["torch.Tensor", np.ndarray]) -> "torch.Tensor":
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self._dtype)
        return torch.from_numpy(np.asarray(data)).to(device=self.device, dtype=self._dtype)
