"""
Optional C++ PagedKVCache bindings (libMLIRLLMRuntime).

Set ``LLMIR_LIB_PATH`` to the shared library, or install a wheel that ships
``libMLIRLLMRuntime.so``. When the library is unavailable, callers fall back
to the NumPy reference implementation in :mod:`llmir.runtime.kv_cache`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import c_bool, c_float, c_int32, c_int64, c_void_p
from typing import Optional, Tuple

import numpy as np
from numpy.ctypeslib import ndpointer

_lib: Optional[ctypes.CDLL] = None


def _find_library() -> Optional[str]:
    explicit = os.environ.get("LLMIR_LIB_PATH")
    if explicit and os.path.exists(explicit):
        return explicit

    search_paths = [
        os.path.join(sys.prefix, "lib"),
        os.path.join(sys.prefix, "lib64"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"),
        "/usr/local/lib",
    ]
    lib_names = [
        "libMLIRLLMRuntime.so",
        "libMLIRLLMRuntime.dylib",
        "MLIRLLMRuntime.dll",
    ]
    for path in search_paths:
        for name in lib_names:
            full_path = os.path.abspath(os.path.join(path, name))
            if os.path.exists(full_path):
                return full_path
    return None


def native_library_available() -> bool:
    """Return True if the native runtime library can be loaded."""
    if os.environ.get("LLMIR_KV_BACKEND", "").lower() == "numpy":
        return False
    try:
        _load_library()
        return True
    except RuntimeError:
        return False


def _load_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = _find_library()
    if path is None:
        raise RuntimeError(
            "LLMIR native library not found. Build the MLIR runtime "
            "(libMLIRLLMRuntime) and set LLMIR_LIB_PATH, or use "
            "LLMIR_KV_BACKEND=numpy for the reference implementation."
        )
    _lib = ctypes.CDLL(path)
    _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    lib.llmir_kvcache_create.argtypes = [
        c_int64,
        c_int64,
        c_int64,
        c_int64,
        c_int64,
        c_int32,
        c_bool,
    ]
    lib.llmir_kvcache_create.restype = c_void_p
    lib.llmir_kvcache_destroy.argtypes = [c_void_p]
    lib.llmir_kvcache_destroy.restype = None
    lib.llmir_kvcache_append.argtypes = [
        c_void_p,
        c_void_p,
        c_void_p,
        c_int32,
        c_int32,
        ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
    ]
    lib.llmir_kvcache_append.restype = c_int32
    lib.llmir_kvcache_lookup.argtypes = [
        c_void_p,
        ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS"),
        ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        c_int32,
        c_void_p,
        c_void_p,
    ]
    lib.llmir_kvcache_lookup.restype = c_int32
    lib.llmir_kvcache_clear_sequence.argtypes = [c_void_p, c_int32]
    lib.llmir_kvcache_clear_sequence.restype = c_int32
    lib.llmir_kvcache_reset.argtypes = [c_void_p]
    lib.llmir_kvcache_reset.restype = None
    lib.llmir_kvcache_get_memory_usage.argtypes = [c_void_p]
    lib.llmir_kvcache_get_memory_usage.restype = c_int64
    lib.llmir_kvcache_get_num_sequences.argtypes = [c_void_p]
    lib.llmir_kvcache_get_num_sequences.restype = c_int32
    if hasattr(lib, "llmir_has_cuda_support"):
        lib.llmir_has_cuda_support.argtypes = []
        lib.llmir_has_cuda_support.restype = c_bool
    if hasattr(lib, "llmir_cuda_runtime_available"):
        lib.llmir_cuda_runtime_available.argtypes = []
        lib.llmir_cuda_runtime_available.restype = c_bool
    if hasattr(lib, "llmir_cuda_device_count"):
        lib.llmir_cuda_device_count.argtypes = []
        lib.llmir_cuda_device_count.restype = c_int32


_DTYPE_MAP = {"float16": 0, "float32": 1, "bfloat16": 2}


class NativePagedKVCacheHandle:
    """Thin ctypes wrapper around one C++ PagedKVCache (single logical layer)."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        block_size: int,
        max_seq_len: int,
        dtype: str,
        enable_gpu: bool,
    ) -> None:
        lib = _load_library()
        dtype_id = _DTYPE_MAP.get(dtype, 0)
        self._handle = lib.llmir_kvcache_create(
            1,
            num_heads,
            head_dim,
            block_size,
            max_seq_len,
            dtype_id,
            enable_gpu,
        )
        if not self._handle:
            raise RuntimeError("llmir_kvcache_create returned NULL")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self._seq_lengths: dict[int, int] = {}

    def close(self) -> None:
        if getattr(self, "_handle", None):
            lib = _load_library()
            lib.llmir_kvcache_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def append(
        self, keys: np.ndarray, values: np.ndarray, seq_ids: np.ndarray
    ) -> np.ndarray:
        lib = _load_library()
        batch_size, seq_len = keys.shape[:2]
        keys = np.ascontiguousarray(keys)
        values = np.ascontiguousarray(values)
        seq_ids = np.ascontiguousarray(seq_ids.astype(np.int32))
        max_blocks = (self.max_seq_len + self.block_size - 1) // self.block_size
        block_indices = np.zeros((batch_size, max_blocks), dtype=np.int32)
        result = lib.llmir_kvcache_append(
            self._handle,
            keys.ctypes.data_as(c_void_p),
            values.ctypes.data_as(c_void_p),
            batch_size,
            seq_len,
            seq_ids,
            block_indices,
        )
        if result != 0:
            raise RuntimeError(f"llmir_kvcache_append failed: {result}")
        for sid in seq_ids:
            self._seq_lengths[int(sid)] = self._seq_lengths.get(int(sid), 0) + seq_len
        return block_indices

    def lookup(
        self, block_indices: np.ndarray, seq_lens: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        lib = _load_library()
        batch_size = len(seq_lens)
        max_seq_len = int(seq_lens.max()) if len(seq_lens) else 0
        block_indices = np.ascontiguousarray(block_indices.astype(np.int32))
        seq_lens = np.ascontiguousarray(seq_lens.astype(np.int32))
        np_dtype = np.float16 if "float16" in self.dtype else np.float32
        output_keys = np.zeros(
            (batch_size, max_seq_len, self.num_heads, self.head_dim), dtype=np_dtype
        )
        output_values = np.zeros_like(output_keys)
        result = lib.llmir_kvcache_lookup(
            self._handle,
            block_indices,
            seq_lens,
            batch_size,
            output_keys.ctypes.data_as(c_void_p),
            output_values.ctypes.data_as(c_void_p),
        )
        if result != 0:
            raise RuntimeError(f"llmir_kvcache_lookup failed: {result}")
        return output_keys, output_values

    def clear_sequence(self, seq_id: int) -> bool:
        lib = _load_library()
        ok = lib.llmir_kvcache_clear_sequence(self._handle, seq_id) == 0
        if ok:
            self._seq_lengths.pop(seq_id, None)
        return ok

    def reset(self) -> None:
        lib = _load_library()
        lib.llmir_kvcache_reset(self._handle)
        self._seq_lengths.clear()

    def get_memory_usage(self) -> int:
        lib = _load_library()
        return int(lib.llmir_kvcache_get_memory_usage(self._handle))

    def get_num_sequences(self) -> int:
        lib = _load_library()
        return int(lib.llmir_kvcache_get_num_sequences(self._handle))

    def get_sequence_length(self, seq_id: int) -> int:
        return self._seq_lengths.get(seq_id, 0)
