// RUN: mlir-opt %s -verify-diagnostics | mlir-opt | FileCheck %s

// CHECK-LABEL: func @test_paged_kv_cache
func.func @test_paged_kv_cache(%query: tensor<1x1x16x64xf16>, 
                               %key: tensor<1x1x16x64xf16>, 
                               %value: tensor<1x1x16x64xf16>, 
                               %seq_ids: tensor<1xi32>) -> tensor<1x1x16x64xf16> {
  
  // Create an empty paged KV cache
  // CHECK: %[[KV_CACHE:.*]] = "llm.paged_kv_cache"
  %kv_cache = "llm.paged_kv_cache"() {
    element_type = f16,
    num_layers = 24,
    num_heads = 16,
    head_dim = 64,
    block_size = 16,
    max_seq_len = 4096
  } : () -> !llm.paged_kv_cache<f16, 24, 16, 64, 16, 4096>
  
  // Append key-value pairs to the KV cache
  // CHECK: %[[NEW_KV:.*]], %[[BLOCK_INDICES:.*]] = llm.append_kv
  %new_kv, %block_indices = llm.append_kv %kv_cache, %key, %value, %seq_ids {
    block_size = 16 : i32,
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f16, 24, 16, 64, 16, 4096>, 
       tensor<1x1x16x64xf16>, tensor<1x1x16x64xf16>, tensor<1xi32>) 
      -> (!llm.paged_kv_cache<f16, 24, 16, 64, 16, 4096>, tensor<1x1xi32>)
  
  // Sequence lengths for attention
  %seq_lens = arith.constant dense<1> : tensor<1xi32>
  
  // Perform paged attention with KV cache
  // CHECK: %[[OUTPUT:.*]] = llm.paged_attention
  %output = llm.paged_attention %query, %new_kv, %block_indices, %seq_lens {
    num_heads = 16 : i32,
    head_dim = 64 : i32,
    scale = 0.125 : f32
  } : (tensor<1x1x16x64xf16>, !llm.paged_kv_cache<f16, 24, 16, 64, 16, 4096>, 
       tensor<1x1xi32>, tensor<1xi32>) -> tensor<1x1x16x64xf16>
  
  // Lookup key-value pairs from the KV cache
  // CHECK: %[[KEYS:.*]], %[[VALUES:.*]] = llm.lookup_kv
  %keys, %values = llm.lookup_kv %new_kv, %block_indices, %seq_lens {
    num_heads = 16 : i32,
    head_dim = 64 : i32
  } : (!llm.paged_kv_cache<f16, 24, 16, 64, 16, 4096>, tensor<1x1xi32>, tensor<1xi32>)
     -> (tensor<1x1x16x64xf16>, tensor<1x1x16x64xf16>)
  
  // Return the attention output
  return %output : tensor<1x1x16x64xf16>
}

// CHECK-LABEL: func @test_multi_sequence_kv_cache
func.func @test_multi_sequence_kv_cache(%query: tensor<2x1x16x64xf16>, 
                                         %key: tensor<2x1x16x64xf16>, 
                                         %value: tensor<2x1x16x64xf16>,
                                         %seq_ids: tensor<2xi32>) -> tensor<2x1x16x64xf16> {
  
  // Create a paged KV cache
  %kv_cache = "llm.paged_kv_cache"() {
    element_type = f16,
    num_layers = 12,
    num_heads = 16,
    head_dim = 64,
    block_size = 16,
    max_seq_len = 2048
  } : () -> !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>
  
  // Append key-value pairs for multiple sequences
  %new_kv, %block_indices = llm.append_kv %kv_cache, %key, %value, %seq_ids {
    block_size = 16 : i32,
    max_seq_len = 2048 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, 
       tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
      -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, tensor<2x1xi32>)
  
  // Sequence lengths
  %seq_lens = arith.constant dense<1> : tensor<2xi32>
  
  // Perform paged attention with multi-sequence batching
  %output = llm.paged_attention %query, %new_kv, %block_indices, %seq_lens {
    num_heads = 16 : i32,
    head_dim = 64 : i32,
    scale = 0.125 : f32
  } : (tensor<2x1x16x64xf16>, !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, 
       tensor<2x1xi32>, tensor<2xi32>) -> tensor<2x1x16x64xf16>
  
  return %output : tensor<2x1x16x64xf16>
} 