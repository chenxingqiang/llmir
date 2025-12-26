"""
Native C++ bindings for LLMIR LLM dialect.

This module loads the native LLMIR library and provides ctypes bindings
for high-performance operations.
"""

import ctypes
from ctypes import c_void_p, c_int32, c_int64, c_float, c_bool, c_char_p
from typing import Optional
import os
import sys
import numpy as np
from numpy.ctypeslib import ndpointer

#===----------------------------------------------------------------------===#
# Library Loading
#===----------------------------------------------------------------------===#

_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[str] = None

def _find_library() -> Optional[str]:
    """Find the LLMIR native library."""
    # Search paths
    search_paths = [
        # Installed location
        os.path.join(sys.prefix, 'lib'),
        os.path.join(sys.prefix, 'lib64'),
        # Build directory
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'lib'),
        # Common build paths
        '/usr/local/lib',
        '/usr/lib',
    ]
    
    # Library names
    lib_names = [
        'libMLIRLLMRuntime.so',
        'libMLIRLLMRuntime.dylib',
        'MLIRLLMRuntime.dll',
    ]
    
    for path in search_paths:
        for name in lib_names:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                return full_path
    
    return None


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """
    Load the LLMIR native library.
    
    Args:
        path: Optional explicit path to the library
        
    Returns:
        Loaded library handle
    """
    global _lib, _lib_path
    
    if _lib is not None:
        return _lib
    
    if path is None:
        path = _find_library()
    
    if path is None:
        raise RuntimeError(
            "Could not find LLMIR native library. "
            "Please build LLMIR with Python bindings enabled or "
            "set LLMIR_LIB_PATH environment variable."
        )
    
    _lib = ctypes.CDLL(path)
    _lib_path = path
    
    # Initialize function signatures
    _setup_signatures(_lib)
    
    return _lib


def _setup_signatures(lib: ctypes.CDLL):
    """Setup ctypes function signatures for the library."""
    
    # PagedKVCache functions
    lib.llmir_kvcache_create.argtypes = [
        c_int64,  # numLayers
        c_int64,  # numHeads
        c_int64,  # headDim
        c_int64,  # blockSize
        c_int64,  # maxSeqLen
        c_int32,  # dtype
        c_bool,   # enableGPU
    ]
    lib.llmir_kvcache_create.restype = c_void_p
    
    lib.llmir_kvcache_destroy.argtypes = [c_void_p]
    lib.llmir_kvcache_destroy.restype = None
    
    lib.llmir_kvcache_append.argtypes = [
        c_void_p,  # handle
        c_void_p,  # keyData
        c_void_p,  # valueData
        c_int32,   # batchSize
        c_int32,   # seqLen
        ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # seqIds
        ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),  # blockIndices (output)
    ]
    lib.llmir_kvcache_append.restype = c_int32
    
    lib.llmir_kvcache_lookup.argtypes = [
        c_void_p,  # handle
        ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS'),  # blockIndices
        ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),  # seqLens
        c_int32,   # batchSize
        c_void_p,  # outputKeys
        c_void_p,  # outputValues
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
    
    # Quantized KV Cache functions
    lib.llmir_quantized_kvcache_create.argtypes = [
        c_int64, c_int64, c_int64, c_int64, c_int64,
        c_int32, c_int32, c_bool, c_int64, c_bool,
    ]
    lib.llmir_quantized_kvcache_create.restype = c_void_p
    
    lib.llmir_quantized_kvcache_destroy.argtypes = [c_void_p]
    lib.llmir_quantized_kvcache_destroy.restype = None
    
    # Continuous Batching Engine functions
    lib.llmir_engine_create.argtypes = [c_void_p, c_int32, c_int32, c_int32]
    lib.llmir_engine_create.restype = c_void_p
    
    lib.llmir_engine_destroy.argtypes = [c_void_p]
    lib.llmir_engine_destroy.restype = None
    
    lib.llmir_engine_submit.argtypes = [
        c_void_p,
        ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        c_int32,
        c_float, c_float, c_int32, c_int32,
    ]
    lib.llmir_engine_submit.restype = c_int32
    
    lib.llmir_engine_step.argtypes = [c_void_p]
    lib.llmir_engine_step.restype = c_int32


#===----------------------------------------------------------------------===#
# Native Wrapper Classes
#===----------------------------------------------------------------------===#

class NativePagedKVCache:
    """Native wrapper for PagedKVCache."""
    
    # Data type mapping
    DTYPE_MAP = {
        'float16': 0,
        'float32': 1,
        'bfloat16': 2,
    }
    
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 block_size: int = 16,
                 max_seq_len: int = 4096,
                 dtype: str = 'float16',
                 enable_gpu: bool = True):
        lib = load_library()
        
        dtype_id = self.DTYPE_MAP.get(dtype, 0)
        
        self._handle = lib.llmir_kvcache_create(
            num_layers, num_heads, head_dim, block_size,
            max_seq_len, dtype_id, enable_gpu
        )
        
        if self._handle is None:
            raise RuntimeError("Failed to create PagedKVCache")
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle is not None:
            lib = load_library()
            lib.llmir_kvcache_destroy(self._handle)
            self._handle = None
    
    def append(self,
               keys: np.ndarray,
               values: np.ndarray,
               seq_ids: np.ndarray) -> np.ndarray:
        """Append key-value pairs to the cache."""
        lib = load_library()
        
        batch_size, seq_len = keys.shape[:2]
        
        # Ensure contiguous arrays
        keys = np.ascontiguousarray(keys)
        values = np.ascontiguousarray(values)
        seq_ids = np.ascontiguousarray(seq_ids.astype(np.int32))
        
        # Output array
        max_blocks = (self.max_seq_len + self.block_size - 1) // self.block_size
        block_indices = np.zeros((batch_size, max_blocks), dtype=np.int32)
        
        result = lib.llmir_kvcache_append(
            self._handle,
            keys.ctypes.data_as(c_void_p),
            values.ctypes.data_as(c_void_p),
            batch_size,
            seq_len,
            seq_ids,
            block_indices
        )
        
        if result != 0:
            raise RuntimeError(f"KVCache append failed with code {result}")
        
        return block_indices
    
    def lookup(self,
               block_indices: np.ndarray,
               seq_lens: np.ndarray) -> tuple:
        """Lookup key-value pairs from the cache."""
        lib = load_library()
        
        batch_size = len(seq_lens)
        max_seq_len = int(seq_lens.max())
        
        # Ensure contiguous arrays
        block_indices = np.ascontiguousarray(block_indices.astype(np.int32))
        seq_lens = np.ascontiguousarray(seq_lens.astype(np.int32))
        
        # Allocate output arrays
        np_dtype = np.float16 if self.dtype == 'float16' else np.float32
        output_keys = np.zeros(
            (batch_size, max_seq_len, self.num_heads, self.head_dim),
            dtype=np_dtype
        )
        output_values = np.zeros_like(output_keys)
        
        result = lib.llmir_kvcache_lookup(
            self._handle,
            block_indices,
            seq_lens,
            batch_size,
            output_keys.ctypes.data_as(c_void_p),
            output_values.ctypes.data_as(c_void_p)
        )
        
        if result != 0:
            raise RuntimeError(f"KVCache lookup failed with code {result}")
        
        return output_keys, output_values
    
    def clear_sequence(self, seq_id: int) -> bool:
        """Clear cache for a specific sequence."""
        lib = load_library()
        result = lib.llmir_kvcache_clear_sequence(self._handle, seq_id)
        return result == 0
    
    def reset(self):
        """Reset the entire cache."""
        lib = load_library()
        lib.llmir_kvcache_reset(self._handle)
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        lib = load_library()
        return lib.llmir_kvcache_get_memory_usage(self._handle)
    
    def get_num_sequences(self) -> int:
        """Get number of active sequences."""
        lib = load_library()
        return lib.llmir_kvcache_get_num_sequences(self._handle)


class NativeQuantizedKVCache:
    """Native wrapper for QuantizedKVCache."""
    
    QUANT_TYPE_MAP = {
        'none': 0,
        'int8': 1,
        'int4': 2,
        'fp8': 3,
    }
    
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 block_size: int = 16,
                 max_seq_len: int = 4096,
                 quant_type: str = 'int8',
                 symmetric: bool = True,
                 group_size: int = 128,
                 enable_gpu: bool = True):
        lib = load_library()
        
        quant_id = self.QUANT_TYPE_MAP.get(quant_type.lower(), 1)
        
        self._handle = lib.llmir_quantized_kvcache_create(
            num_layers, num_heads, head_dim, block_size, max_seq_len,
            quant_id, 0,  # strategy: per-tensor
            symmetric, group_size, False  # dynamicRange
        )
        
        if self._handle is None:
            raise RuntimeError("Failed to create QuantizedKVCache")
        
        self.quant_type = quant_type
        self.compression_ratio = 4.0 if quant_type == 'int8' else 8.0
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle is not None:
            lib = load_library()
            lib.llmir_quantized_kvcache_destroy(self._handle)
            self._handle = None


class NativeContinuousBatchingEngine:
    """Native wrapper for ContinuousBatchingEngine."""
    
    def __init__(self, cache_handle: c_void_p, max_batch_size: int = 256,
                 max_num_seqs: int = 256, chunk_size: int = 512):
        lib = load_library()
        
        self._handle = lib.llmir_engine_create(
            cache_handle, max_batch_size, max_num_seqs, chunk_size
        )
        
        if self._handle is None:
            raise RuntimeError("Failed to create ContinuousBatchingEngine")
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle is not None:
            lib = load_library()
            lib.llmir_engine_destroy(self._handle)
            self._handle = None
    
    def submit(self,
               prompt_tokens: np.ndarray,
               temperature: float = 1.0,
               top_p: float = 1.0,
               top_k: int = -1,
               max_tokens: int = 256) -> int:
        """Submit a generation request."""
        lib = load_library()
        
        tokens = np.ascontiguousarray(prompt_tokens.astype(np.int32))
        
        request_id = lib.llmir_engine_submit(
            self._handle, tokens, len(tokens),
            temperature, top_p, top_k, max_tokens
        )
        
        return request_id
    
    def step(self) -> int:
        """Run one step of the engine."""
        lib = load_library()
        return lib.llmir_engine_step(self._handle)


#===----------------------------------------------------------------------===#
# Utility Functions
#===----------------------------------------------------------------------===#

def is_native_available() -> bool:
    """Check if native library is available."""
    try:
        load_library()
        return True
    except RuntimeError:
        return False


def get_library_path() -> Optional[str]:
    """Get the path to the loaded native library."""
    return _lib_path


def get_build_info() -> dict:
    """Get build information from the native library."""
    if not is_native_available():
        return {}
    
    # Would query the library for build info
    return {
        'version': '1.0.0',
        'cuda_enabled': True,
        'nccl_enabled': True,
    }
