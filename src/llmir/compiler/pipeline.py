"""High-level compile + reference-run pipeline (P2 MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from llmir.compiler.interpreter import run_kv_micro_pipeline_reference
from llmir.compiler.kv_emit import KVMicroPipelineConfig, emit_kv_micro_pipeline_mlir
from llmir.compiler.opt_driver import OptResult, run_mlir_opt


@dataclass
class CompilePipelineResult:
    """Artifacts from ``compile_kv_micro_pipeline``."""

    mlir: str
    opt: Optional[OptResult] = None
    reference_output: Optional[np.ndarray] = None
    reference_backend: str = ""
    torch_max_abs_diff: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compile_kv_micro_pipeline(
    cfg: KVMicroPipelineConfig,
    *,
    run_opt: bool = True,
    run_reference: bool = False,
    compare_torch: bool = False,
    prefer_native: Optional[bool] = None,
    seed: int = 0,
) -> CompilePipelineResult:
    """
    Emit MLIR, optionally lower with mlir-opt, optionally run Python reference.

    When ``compare_torch`` is True, compares reference output to
    ``torch.nn.functional.scaled_dot_product_attention`` (requires torch).
    """
    mlir = emit_kv_micro_pipeline_mlir(cfg)
    result = CompilePipelineResult(mlir=mlir)

    if run_opt:
        result.opt = run_mlir_opt(mlir)

    if not run_reference and not compare_torch:
        return result

    rng = np.random.default_rng(seed)
    b, s, h, d = cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.head_dim
    np_dtype = np.float32 if cfg.dtype == "f32" else np.float16
    keys = rng.standard_normal((b, s, h, d)).astype(np_dtype)
    values = rng.standard_normal((b, s, h, d)).astype(np_dtype)
    # Query = last token (decode-step semantics after prefill append).
    query = keys[:, -1:, :, :].copy()

    out, backend = run_kv_micro_pipeline_reference(
        keys, values, query, cfg=cfg, prefer_native=prefer_native
    )
    result.reference_output = out
    result.reference_backend = backend

    if compare_torch:
        import torch

        # Decode-step semantics: Q length 1, KV length S. Match explicit einsum
        # (PyTorch SDPA differs when Q/K sequence lengths differ).
        scale = 1.0 / (cfg.head_dim**0.5)
        tq = torch.from_numpy(query)
        tk = torch.from_numpy(keys)
        tv = torch.from_numpy(values)
        with torch.no_grad():
            scores = torch.einsum("bqhd,bkhd->bqhk", tq, tk) * scale
            attn = torch.softmax(scores, dim=-1)
            torch_out = torch.einsum("bqhk,bkhd->bqhd", attn, tv)
        diff = float(np.max(np.abs(out - torch_out.numpy())))
        result.torch_max_abs_diff = diff
        result.metadata["torch_allclose_1e-4"] = diff < 1e-4
        result.metadata["torch_allclose_1e-3"] = diff < 1e-3

    return result
