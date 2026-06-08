// MVP-A: single-layer KV micro-pipeline (paper Section 3–4).
// RUN: mlir-opt %s -llm-optimize-kv-cache | FileCheck %s --check-prefix=OPT
// RUN: mlir-opt %s -llm-optimize-kv-cache -llm-lower-kv-cache-ops | FileCheck %s --check-prefix=LOWER

// OPT-LABEL: func @mvp_single_layer
// OPT: llm.append_kv
// OPT-SAME: block_size = 32
// OPT: llm.paged_attention

// LOWER-LABEL: func @mvp_single_layer
// LOWER: call @mlir_llm_append_kv
// LOWER: call @mlir_llm_paged_attention

func.func @mvp_single_layer(
    %cache: !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 128>,
    %keys: tensor<1x4x4x8xf32>,
    %values: tensor<1x4x4x8xf32>,
    %seq_ids: tensor<1xi32>,
    %query: tensor<1x1x4x8xf32>
) -> tensor<1x1x4x8xf32> {
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 1024 : i32,
    max_seq_len = 128 : i32
  } : (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 128>, tensor<1x4x4x8xf32>, tensor<1x4x4x8xf32>, tensor<1xi32>)
      -> (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 128>, tensor<1x1xi32>)
  %seq_lens = arith.constant dense<4> : tensor<1xi32>
  %out = llm.paged_attention %query, %new_cache, %block_indices, %seq_lens {
    num_heads = 4 : i32,
    head_dim = 8 : i32,
    scale = 3.53553391e-01 : f32
  } : (tensor<1x1x4x8xf32>, !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 128>, tensor<1x1xi32>, tensor<1xi32>)
      -> tensor<1x1x4x8xf32>
  return %out : tensor<1x1x4x8xf32>
}
