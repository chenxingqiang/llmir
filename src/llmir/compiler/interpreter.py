"""Reference interpreter for LLM KV micro-pipeline semantics (Python/NumPy)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from llmir.compiler.kv_emit import KVMicroPipelineConfig
from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_factory import create_paged_kv_cache


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return np.asarray(e / np.sum(e, axis=axis, keepdims=True))


def numpy_paged_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """
    Reference paged attention for a single query step (Q=1).

    ``query``: [batch, 1, heads, head_dim]
    ``keys`` / ``values``: [batch, seq_len, heads, head_dim]
    """
    q = query.astype(np.float64)
    k = keys.astype(np.float64)
    v = values.astype(np.float64)
    scores = np.einsum("bqhd,bkhd->bqhk", q, k) * scale
    attn = _softmax(scores, axis=-1)
    out = np.einsum("bqhk,bkhd->bqhd", attn, v)
    return np.asarray(out.astype(query.dtype, copy=False))


def run_kv_micro_pipeline_reference(
    keys: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    *,
    cfg: KVMicroPipelineConfig | None = None,
    prefer_native: bool | None = None,
) -> Tuple[np.ndarray, str]:
    """
    Execute append → lookup → paged_attention using ``create_paged_kv_cache``.

    Returns ``(output, backend_name)``.
    """
    cfg = cfg or KVMicroPipelineConfig(
        batch_size=keys.shape[0],
        seq_len=keys.shape[1],
        num_heads=keys.shape[2],
        head_dim=keys.shape[3],
    )
    kv_config = KVCacheConfig(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        block_size=cfg.block_size,
        max_seq_len=cfg.max_seq_len,
        dtype="float32" if cfg.dtype == "f32" else "float16",
        enable_gpu=False,
    )
    from llmir.runtime.kv_factory import kv_cache_backend_name

    cache = create_paged_kv_cache(kv_config, prefer_native=prefer_native)
    backend = kv_cache_backend_name(cache)
    batch = keys.shape[0]
    seq_ids = np.arange(batch, dtype=np.int32)
    block_indices = cache.append(keys, values, seq_ids)
    seq_lens = np.full(batch, keys.shape[1], dtype=np.int32)
    k_cached, v_cached = cache.lookup(block_indices, seq_lens)
    scale = 1.0 / (cfg.head_dim**0.5)
    out = numpy_paged_attention(query, k_cached, v_cached, scale=scale)
    return out, backend
