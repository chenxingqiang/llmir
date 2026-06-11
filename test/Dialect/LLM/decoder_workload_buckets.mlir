// E1 block-size rewrite on S1/S2/S3 decoder workload trace shapes.
// RUN: mlir-opt %s -llm-optimize-kv-cache -split-input-file | FileCheck %s

// CHECK-LABEL: func @bucket_s1_short_multitenant
// CHECK: llm.append_kv
// CHECK-SAME: block_size = 32
func.func @bucket_s1_short_multitenant(
    %cache: !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 256>,
    %keys: tensor<1x16x4x8xf32>,
    %values: tensor<1x16x4x8xf32>,
    %seq_ids: tensor<1xi32>
) -> !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 256> {
  // L_s=128, L_u=16 (S1 short multi-tenant)
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 1024 : i32,
    max_seq_len = 256 : i32
  } : (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 256>, tensor<1x16x4x8xf32>, tensor<1x16x4x8xf32>, tensor<1xi32>)
      -> (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 256>, tensor<1x1xi32>)
  return %new_cache : !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 256>
}

// -----

// CHECK-LABEL: func @bucket_s2_rag_shared_system
// CHECK: llm.append_kv
// CHECK-SAME: block_size = 32
func.func @bucket_s2_rag_shared_system(
    %cache: !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 4096>,
    %keys: tensor<1x8x4x8xf32>,
    %values: tensor<1x8x4x8xf32>,
    %seq_ids: tensor<1xi32>
) -> !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 4096> {
  // L_s=2048, L_u=8 (S2 RAG shared system)
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 1024 : i32,
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 4096>, tensor<1x8x4x8xf32>, tensor<1x8x4x8xf32>, tensor<1xi32>)
      -> (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 4096>, tensor<1x1xi32>)
  return %new_cache : !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 4096>
}

// -----

// CHECK-LABEL: func @bucket_s3_long_document
// CHECK: llm.append_kv
// CHECK-SAME: block_size = 32
func.func @bucket_s3_long_document(
    %cache: !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 16384>,
    %keys: tensor<1x64x4x8xf32>,
    %values: tensor<1x64x4x8xf32>,
    %seq_ids: tensor<1xi32>
) -> !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 16384> {
  // L_s=8192, L_u=64 (S3 long-document prefill)
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 1024 : i32,
    max_seq_len = 16384 : i32
  } : (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 16384>, tensor<1x64x4x8xf32>, tensor<1x64x4x8xf32>, tensor<1xi32>)
      -> (!llm.paged_kv_cache<f32, 1, 4, 8, 1024, 16384>, tensor<1x1xi32>)
  return %new_cache : !llm.paged_kv_cache<f32, 1, 4, 8, 1024, 16384>
}
