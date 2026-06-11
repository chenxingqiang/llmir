"""Minimal PyTorch module for P2 import / MLIR emit smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from llmir.compiler.kv_emit import KVMicroPipelineConfig, emit_kv_micro_pipeline_mlir
from llmir.importers.pytorch import ImportConfig, ImportMode, PyTorchImporter


@dataclass
class ToyAttentionSpec:
    """Shape metadata for a toy SDPA module."""

    batch_size: int = 1
    seq_len: int = 4
    num_heads: int = 4
    head_dim: int = 8


def build_toy_sdpa_module(spec: Optional[ToyAttentionSpec] = None):
    """
    Return a tiny ``nn.Module`` that calls ``scaled_dot_product_attention``.

    Requires ``torch``.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    spec = spec or ToyAttentionSpec()
    d_model = spec.num_heads * spec.head_dim

    class ToySDPA(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, s, _ = x.shape
            x = x.view(b, s, spec.num_heads, spec.head_dim)
            return F.scaled_dot_product_attention(x, x, x, is_causal=True)

    return ToySDPA(), d_model


def import_toy_attention_to_mlir(
    spec: Optional[ToyAttentionSpec] = None,
    *,
    enable_kv_cache: bool = True,
) -> str:
    """
    Trace toy SDPA and emit MLIR via :class:`MLIRBuilder`.

    Falls back to :func:`emit_kv_micro_pipeline_mlir` when FX finds no patterns.
    """
    import torch

    spec = spec or ToyAttentionSpec()
    module, d_model = build_toy_sdpa_module(spec)
    module.eval()
    sample = torch.randn(spec.batch_size, spec.seq_len, d_model)
    importer = PyTorchImporter(
        ImportConfig(
            mode=ImportMode.FX,
            enable_kv_cache=enable_kv_cache,
            batch_size=spec.batch_size,
            seq_length=spec.seq_len,
            dtype="fp32",
            device="cpu",
        )
    )
    model_config = {
        "hidden_size": d_model,
        "num_layers": 1,
        "num_attention_heads": spec.num_heads,
        "head_dim": spec.head_dim,
    }
    mlir = importer.import_model(
        module, sample_inputs={"x": sample}, model_config=model_config
    )
    if "llm.paged_attention" not in mlir and "llm.attention" not in mlir:
        kv_cfg = KVMicroPipelineConfig(
            batch_size=spec.batch_size,
            seq_len=spec.seq_len,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            dtype="f32",
        )
        mlir = emit_kv_micro_pipeline_mlir(kv_cfg)
    return mlir


def toy_attention_model_config(
    spec: Optional[ToyAttentionSpec] = None,
) -> Dict[str, Any]:
    spec = spec or ToyAttentionSpec()
    return {
        "num_layers": 1,
        "num_attention_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "hidden_size": spec.num_heads * spec.head_dim,
    }
