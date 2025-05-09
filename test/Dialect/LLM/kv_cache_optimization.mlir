// RUN: mlir-opt %s -llm-optimize-kv-cache -split-input-file | FileCheck %s

// Test block size optimization for append_kv operations

// CHECK-LABEL: test_block_size_optimization
func.func @test_block_size_optimization(%cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, 
                                        %keys: tensor<2x1x16x64xf16>, 
                                        %values: tensor<2x1x16x64xf16>, 
                                        %seq_ids: tensor<2xi32>) -> !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096> {
  // CHECK: %{{.*}}, %{{.*}} = llm.append_kv %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {block_size = 256 : i32, max_seq_len = 4096 : i32}
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 1024 : i32, // Too large block size, should be optimized
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
      -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>)
  
  return %new_cache : !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>
}

// -----

// Test that small block sizes are left unchanged

// CHECK-LABEL: test_small_block_size_unchanged
func.func @test_small_block_size_unchanged(%cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, 
                                           %keys: tensor<2x1x16x64xf16>, 
                                           %values: tensor<2x1x16x64xf16>, 
                                           %seq_ids: tensor<2xi32>) -> !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096> {
  // CHECK: %{{.*}}, %{{.*}} = llm.append_kv %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {block_size = 16 : i32, max_seq_len = 4096 : i32}
  %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 16 : i32, // Already small enough, should be unchanged
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
      -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>)
  
  return %new_cache : !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>
}

// -----

// Test compatibility checking between append_kv and lookup_kv operations

// CHECK-LABEL: test_operation_compatibility
func.func @test_operation_compatibility(%cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, 
                                       %keys: tensor<2x1x16x64xf16>, 
                                       %values: tensor<2x1x16x64xf16>, 
                                       %seq_ids: tensor<2xi32>,
                                       %block_indices: tensor<2x128xi32>,
                                       %seq_lens: tensor<2xi32>) 
                                       -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>) {
  %new_cache, %new_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
    block_size = 16 : i32,
    max_seq_len = 4096 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1x16x64xf16>, tensor<2x1x16x64xf16>, tensor<2xi32>) 
      -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x1xi32>)
  
  // CHECK: %{{.*}}, %{{.*}} = llm.lookup_kv %{{.*}}, %{{.*}}, %{{.*}} {head_dim = 64 : i32, num_heads = 16 : i32}
  %lookup_keys, %lookup_values = llm.lookup_kv %new_cache, %block_indices, %seq_lens {
    num_heads = 16 : i32, // Should match the KV cache type
    head_dim = 64 : i32
  } : (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 4096>, tensor<2x128xi32>, tensor<2xi32>) 
      -> (tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>)
  
  return %lookup_keys, %lookup_values : tensor<2x128x16x64xf16>, tensor<2x128x16x64xf16>
} 