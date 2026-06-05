"""Emit valid LLM dialect MLIR for KV-cache + paged-attention micro-pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVMicroPipelineConfig:
    """Concrete shapes for a single-layer KV micro-pipeline."""

    num_layers: int = 1
    num_heads: int = 4
    head_dim: int = 8
    block_size: int = 16
    max_seq_len: int = 128
    batch_size: int = 1
    seq_len: int = 4
    dtype: str = "f32"  # mlir element type suffix: f16 or f32


def emit_kv_micro_pipeline_mlir(cfg: KVMicroPipelineConfig) -> str:
    """
    Emit MLIR matching ``test/Dialect/LLM/kv_cache_ops.mlir`` structure.

    The module defines ``@kv_micro_pipeline`` which append → lookup → paged_attention.
    """
    dt = cfg.dtype
    b, s, h, d = cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.head_dim
    scale = 1.0 / (d**0.5)
    cache_ty = (
        f"!llm.paged_kv_cache<{dt}, {cfg.num_layers}, {h}, {d}, "
        f"{cfg.block_size}, {cfg.max_seq_len}>"
    )
    return f"""module {{
  func.func @kv_micro_pipeline(
      %cache: {cache_ty},
      %keys: tensor<{b}x{s}x{h}x{d}x{dt}>,
      %values: tensor<{b}x{s}x{h}x{d}x{dt}>,
      %seq_ids: tensor<{b}xi32>,
      %query: tensor<{b}x1x{h}x{d}x{dt}>
  ) -> tensor<{b}x1x{h}x{d}x{dt}> {{
    %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {{
      block_size = {cfg.block_size} : i32,
      max_seq_len = {cfg.max_seq_len} : i32
    }} : ({cache_ty}, tensor<{b}x{s}x{h}x{d}x{dt}>, tensor<{b}x{s}x{h}x{d}x{dt}>, tensor<{b}xi32>)
        -> ({cache_ty}, tensor<{b}x1xi32>)
    %seq_lens = arith.constant dense<{s}> : tensor<{b}xi32>
    %k, %v = llm.lookup_kv %new_cache, %block_indices, %seq_lens {{
      num_heads = {h} : i32,
      head_dim = {d} : i32
    }} : ({cache_ty}, tensor<{b}x1xi32>, tensor<{b}xi32>)
        -> (tensor<{b}x{s}x{h}x{d}x{dt}>, tensor<{b}x{s}x{h}x{d}x{dt}>)
    %out = llm.paged_attention %query, %new_cache, %block_indices, %seq_lens {{
      num_heads = {h} : i32,
      head_dim = {d} : i32,
      scale = {scale:.8f} : f32
    }} : (tensor<{b}x1x{h}x{d}x{dt}>, {cache_ty}, tensor<{b}x1xi32>, tensor<{b}xi32>)
        -> tensor<{b}x1x{h}x{d}x{dt}>
    return %out : tensor<{b}x1x{h}x{d}x{dt}>
  }}
}}
"""
