// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// Test the basic KV cache operations defined in the LLM dialect

// CHECK-LABEL: test_append_kv
func.func @test_append_kv(%cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, 
                         %keys: tensor<2x1x16x64xf16>, 
                         %values: tensor<2x1x16x64xf16>, 
                         %seq_ids: tensor<2xi32>) -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>) {
  // CHECK: %{{.*}}, %{{.*}} = llm.append_kv %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {block_size = 16 : i32, max_seq_len = 4096 : i32} : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>)
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 16 : i32,
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
      -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>)
  
  return %new_cache, %block_indices : !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>
}

// -----

// CHECK-LABEL: test_lookup_kv
func.func @test_lookup_kv(%cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>,
                         %block_indices: tensor<2x128xi32>,
                         %seq_lens: tensor<2xi32>) -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>) {
  // CHECK: %{{.*}}, %{{.*}} = llm.lookup_kv %{{.*}}, %{{.*}}, %{{.*}} {head_dim = 64 : i32, num_heads = 16 : i32} : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x128xi32>, tensor<2xi32>) -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>)
  %keys, %values = llm.lookup_kv %cache, %block_indices, %seq_lens {
    num_heads = 16 : i32,
    head_dim = 64 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x128xi32>, tensor<2xi32>) 
      -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>)
  
  return %keys, %values : tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>
}

// -----

// CHECK-LABEL: test_paged_attention
func.func @test_paged_attention(%query: tensor<2x1x16x64xf16>,
                               %cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>,
                               %block_indices: tensor<2x128xi32>,
                               %seq_lens: tensor<2xi32>) -> tensor<2x1x16x64xf16> {
  // CHECK: %{{.*}} = llm.paged_attention %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {head_dim = 64 : i32, num_heads = 16 : i32, scale = 1.250000e-01 : f32} : (tensor<2x1x16x64xf16>, !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x128xi32>, tensor<2xi32>) -> tensor<2x1x16x64xf16>
  %output = llm.paged_attention %query, %cache, %block_indices, %seq_lens {
    num_heads = 16 : i32,
    head_dim = 64 : i32,
    scale = 0.125 : f32
  } : (tensor<2x1x16x64xf16>, !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x128xi32>, tensor<2xi32>) 
      -> tensor<2x1x16x64xf16>
  
  return %output : tensor<2x1x16x64xf16>
}

// -----

// CHECK-LABEL: test_full_kv_cache_pipeline
func.func @test_full_kv_cache_pipeline(%cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>,
                                      %keys: tensor<1x1x16x64xf16>,
                                      %values: tensor<1x1x16x64xf16>,
                                      %seq_ids: tensor<1xi32>,
                                      %query: tensor<1x1x16x64xf16>) -> tensor<1x1x16x64xf16> {
  // Append new token to the cache
  // CHECK: %[[NEW_CACHE:.*]], %[[BLOCK_INDICES:.*]] = llm.append_kv
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 16 : i32,
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<1x1x16x64xf16>, tensor<1x1x16x64xf16>, tensor<1xi32>) 
      -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<1x1xi32>)
  
  // Convert single token indices to sequence length indices
  %seq_length = arith.constant dense<1> : tensor<1xi32>
  
  // Pad block indices to match sequence length
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %seq_indices = tensor.from_elements %c0 : tensor<1xindex>
  %seq_block_indices = tensor.generate %c1, %c1 {
  ^bb0(%i: index, %j: index):
    %extract = tensor.extract %block_indices[%i, %j] : tensor<1x1xi32>
    tensor.yield %extract : i32
  } : tensor<1x1xi32>
  
  // Perform paged attention with the updated cache
  // CHECK: %[[OUTPUT:.*]] = llm.paged_attention %{{.*}}, %[[NEW_CACHE]]
  %output = llm.paged_attention %query, %new_cache, %seq_block_indices, %seq_length {
    num_heads = 16 : i32,
    head_dim = 64 : i32,
    scale = 0.125 : f32
  } : (tensor<1x1x16x64xf16>, !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<1x1xi32>, tensor<1xi32>) 
      -> tensor<1x1x16x64xf16>
  
  return %output : tensor<1x1x16x64xf16>
} 