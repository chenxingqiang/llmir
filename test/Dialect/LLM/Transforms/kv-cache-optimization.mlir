// RUN: mlir-opt %s -llm-optimize-kv-cache | FileCheck %s

// TEST: Optimize block size for AppendKVOp
// CHECK-LABEL: func @test_optimize_block_size
func.func @test_optimize_block_size(%keys: tensor<2x16x16x64xf16>, %values: tensor<2x16x16x64xf16>, 
                      %seq_ids: tensor<2xi32>, %kv_cache: !llm.kvcache) 
                      -> (!llm.kvcache, tensor<2x16xi32>) {
  // The original block size is 256, which is too large for a sequence length of 16
  // CHECK: llm.append_kv
  // CHECK-SAME: block_size = 16
  %kv_cache_updated, %block_indices = llm.append_kv(%keys, %values, %seq_ids, %kv_cache) {
    block_size = 256 : i32,
    max_seq_len = 2048 : i32
  } : (tensor<2x16x16x64xf16>, tensor<2x16x16x64xf16>, tensor<2xi32>, !llm.kvcache) -> 
    (!llm.kvcache, tensor<2x16xi32>)
  return %kv_cache_updated, %block_indices : !llm.kvcache, tensor<2x16xi32>
}

// TEST: Add scale to PagedAttentionOp that doesn't have one
// CHECK-LABEL: func @test_add_scale_to_attention
func.func @test_add_scale_to_attention(%query: tensor<2x8x16x64xf16>, 
                           %block_indices: tensor<2x128xi32>, 
                           %seq_lens: tensor<2xi32>,
                           %kv_cache: !llm.kvcache) 
                           -> tensor<2x8x16x64xf16> {
  // Original op has no scale
  // CHECK: llm.paged_attention
  // CHECK-SAME: scale = 1.250000e-01
  %output = llm.paged_attention(%query, %block_indices, %seq_lens, %kv_cache) {
    num_heads = 16 : i32,
    head_dim = 64 : i32
  } : (tensor<2x8x16x64xf16>, tensor<2x128xi32>, tensor<2xi32>, !llm.kvcache) 
      -> tensor<2x8x16x64xf16>
  return %output : tensor<2x8x16x64xf16>
}

// TEST: Fuse duplicate KV cache operations
// CHECK-LABEL: func @test_fuse_duplicate_ops
func.func @test_fuse_duplicate_ops(%keys1: tensor<2x8x16x64xf16>, %values1: tensor<2x8x16x64xf16>, 
                       %keys2: tensor<2x8x16x64xf16>, %values2: tensor<2x8x16x64xf16>,
                       %seq_ids: tensor<2xi32>, %kv_cache: !llm.kvcache) 
                       -> (!llm.kvcache, tensor<2x8xi32>) {
  // First append operation
  // CHECK: llm.append_kv
  %kv_cache1, %block_indices1 = llm.append_kv(%keys1, %values1, %seq_ids, %kv_cache) {
    block_size = 64 : i32,
    max_seq_len = 2048 : i32
  } : (tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>, tensor<2xi32>, !llm.kvcache) -> 
    (!llm.kvcache, tensor<2x8xi32>)
    
  // Second append operation with the same KV cache should be optimized away
  // CHECK-NOT: llm.append_kv
  %kv_cache2, %block_indices2 = llm.append_kv(%keys2, %values2, %seq_ids, %kv_cache1) {
    block_size = 32 : i32,
    max_seq_len = 2048 : i32
  } : (tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>, tensor<2xi32>, !llm.kvcache) -> 
    (!llm.kvcache, tensor<2x8xi32>)
    
  // CHECK: return
  return %kv_cache2, %block_indices2 : !llm.kvcache, tensor<2x8xi32>
} 