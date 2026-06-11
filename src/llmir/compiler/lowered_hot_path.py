"""M5: execute KV micro-pipeline via lowered ``mlir_llm_*`` call sequence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llmir.compiler.block_size import optimize_block_size_attr
from llmir.compiler.interpreter import numpy_paged_attention
from llmir.compiler.kv_emit import KVMicroPipelineConfig, emit_kv_micro_pipeline_mlir
from llmir.compiler.opt_driver import find_mlir_opt, run_mlir_opt
from llmir.compiler.pipeline import compile_kv_micro_pipeline
from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_factory import create_paged_kv_cache, kv_cache_backend_name

LOWERED_RUNTIME_SYMBOLS: Tuple[str, ...] = (
    "mlir_llm_append_kv",
    "mlir_llm_lookup_kv",
    "mlir_llm_paged_attention",
)


@dataclass
class LoweredHotPathResult:
    """M5 verification payload."""

    experiment: str = "M5"
    mode: str = "lowered_hot_path"
    mlir_lowered: bool = False
    lowered_symbols_present: Dict[str, bool] = field(default_factory=dict)
    execution_path: str = ""
    reference_backend: str = ""
    max_abs_diff_vs_reference: Optional[float] = None
    matches_reference: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_lowered_mlir(lowered_mlir: str) -> Dict[str, bool]:
    """Check that lowered IR contains expected runtime call symbols."""
    return {symbol: symbol in lowered_mlir for symbol in LOWERED_RUNTIME_SYMBOLS}


def _kv_config_from_micro(cfg: KVMicroPipelineConfig) -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        block_size=cfg.block_size,
        max_seq_len=cfg.max_seq_len,
        dtype="float32" if cfg.dtype == "f32" else "float16",
        enable_gpu=False,
    )


def execute_semantic_lowered_hot_path(
    keys: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    *,
    cfg: KVMicroPipelineConfig,
) -> Tuple[np.ndarray, str]:
    """
    Run append_kv → lookup_kv → paged_attention in lowered-op order.

    Maps to ``mlir_llm_append_kv``, ``mlir_llm_lookup_kv``, and attention
    (reference kernel until ``mlir_llm_paged_attention`` is production-grade).
    """
    kv_config = _kv_config_from_micro(cfg)
    cache = create_paged_kv_cache(kv_config, device="cpu")
    backend = kv_cache_backend_name(cache)
    batch = keys.shape[0]
    seq_ids = np.arange(batch, dtype=np.int32)
    block_indices = cache.append(keys, values, seq_ids)
    seq_lens = np.full(batch, keys.shape[1], dtype=np.int32)
    k_cached, v_cached = cache.lookup(block_indices, seq_lens)
    scale = 1.0 / (cfg.head_dim**0.5)
    out = numpy_paged_attention(query, k_cached, v_cached, scale=scale)
    return out, f"semantic_lowered::{backend}"


def lower_kv_micro_pipeline_mlir(
    cfg: KVMicroPipelineConfig,
    *,
    oversized_block_size: int = 1024,
) -> Tuple[str, str, int]:
    """Emit MLIR, apply block-size pass, lower to runtime calls. Returns (before, lowered, block_size)."""
    emit_cfg = KVMicroPipelineConfig(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        block_size=oversized_block_size,
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        dtype=cfg.dtype,
    )
    mlir_before = emit_kv_micro_pipeline_mlir(emit_cfg)
    optimal = optimize_block_size_attr(oversized_block_size, [cfg.seq_len])
    mlir_optimized = mlir_before.replace(
        f"block_size = {oversized_block_size} : i32",
        f"block_size = {optimal} : i32",
        1,
    )
    opt = run_mlir_opt(
        mlir_optimized,
        passes=("-llm-optimize-kv-cache", "-llm-lower-kv-cache-ops"),
    )
    if opt.success:
        return mlir_before, opt.stdout, optimal
    return mlir_before, "", optimal


def run_lowered_hot_path_verification(
    cfg: Optional[KVMicroPipelineConfig] = None,
    *,
    seed: int = 42,
    oversized_block_size: int = 1024,
    atol: float = 1e-5,
) -> LoweredHotPathResult:
    """Full M5: lower (if mlir-opt present) + semantic hot path vs compile reference."""
    cfg = cfg or KVMicroPipelineConfig(seq_len=8, num_heads=2, head_dim=16)
    tuned = KVMicroPipelineConfig(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        block_size=optimize_block_size_attr(oversized_block_size, [cfg.seq_len]),
        max_seq_len=cfg.max_seq_len,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        dtype=cfg.dtype,
    )

    mlir_before, lowered, block_after = lower_kv_micro_pipeline_mlir(
        cfg, oversized_block_size=oversized_block_size
    )
    symbols = verify_lowered_mlir(lowered) if lowered else {s: False for s in LOWERED_RUNTIME_SYMBOLS}
    mlir_lowered = bool(lowered) and all(symbols.values())

    ref = compile_kv_micro_pipeline(
        tuned,
        run_opt=False,
        run_reference=True,
        compare_torch=False,
        seed=seed,
    )
    assert ref.reference_output is not None

    rng = np.random.default_rng(seed)
    b, s, h, d = tuned.batch_size, tuned.seq_len, tuned.num_heads, tuned.head_dim
    np_dtype = np.float32 if tuned.dtype == "f32" else np.float16
    keys = rng.standard_normal((b, s, h, d)).astype(np_dtype)
    values = rng.standard_normal((b, s, h, d)).astype(np_dtype)
    query = keys[:, -1:, :, :].copy()

    hot_out, exec_path = execute_semantic_lowered_hot_path(
        keys, values, query, cfg=tuned
    )
    diff = float(np.max(np.abs(hot_out - ref.reference_output)))
    matches = diff < atol

    return LoweredHotPathResult(
        mlir_lowered=mlir_lowered,
        lowered_symbols_present=symbols,
        execution_path=exec_path,
        reference_backend=ref.reference_backend,
        max_abs_diff_vs_reference=diff,
        matches_reference=matches,
        config={
            "block_size_after_pass": block_after,
            "seq_len": tuned.seq_len,
            "num_heads": tuned.num_heads,
            "head_dim": tuned.head_dim,
        },
        metadata={
            "mlir_opt_available": find_mlir_opt() is not None,
            "lowered_mlir_bytes": len(lowered),
            "atol": atol,
            "has_mlir_before": bool(mlir_before),
        },
    )
