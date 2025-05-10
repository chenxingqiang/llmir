// RUN: mlir-opt %s -llm-lower-kv-cache-ops | FileCheck %s

// CHECK-LABEL: func @test_append_kv
func.func @test_append_kv(%keys: tensor<2x8x16x64xf16>, %values: tensor<2x8x16x64xf16>, 
                      %seq_ids: tensor<2xi32>, %kv_cache: !llm.kvcache) 
                      -> (!llm.kvcache, tensor<2x8xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<2x8x16x64xf16>
  // CHECK: %[[DIM1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<2x8x16x64xf16>
  // CHECK: %[[RESULT:.*]]:2 = call @mlir_llm_append_kv(%arg3, %arg0, %arg1, %arg2, %[[DIM0]], %[[DIM1]]) 
  // CHECK-SAME: : (!llm.kvcache, tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>, tensor<2xi32>, index, index) -> (!llm.kvcache, tensor<2x8xi32>)
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1
  %kv_cache_updated, %block_indices = llm.append_kv(%keys, %values, %seq_ids, %kv_cache) : 
    (tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>, tensor<2xi32>, !llm.kvcache) -> 
    (!llm.kvcache, tensor<2x8xi32>)
  return %kv_cache_updated, %block_indices : !llm.kvcache, tensor<2x8xi32>
}

// CHECK-LABEL: func @test_lookup_kv
func.func @test_lookup_kv(%block_indices: tensor<2x8xi32>, %seq_lens: tensor<2xi32>, 
                      %kv_cache: !llm.kvcache) 
                      -> (tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<2x8xi32>
  // CHECK: %[[DIM1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<2x8xi32>
  // CHECK: %[[RESULT:.*]]:2 = call @mlir_llm_lookup_kv(%arg2, %arg0, %arg1, %[[DIM0]], %[[DIM1]])
  // CHECK-SAME: : (!llm.kvcache, tensor<2x8xi32>, tensor<2xi32>, index, index) -> (tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>)
  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1
  %keys, %values = llm.lookup_kv(%block_indices, %seq_lens, %kv_cache) : 
    (tensor<2x8xi32>, tensor<2xi32>, !llm.kvcache) -> 
    (tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>)
  return %keys, %values : tensor<2x8x16x64xf16>, tensor<2x8x16x64xf16>
}

// CHECK-LABEL: func @test_paged_attention
func.func @test_paged_attention(%query: tensor<2x8x16x64xf16>, 
                           %block_indices: tensor<2x128xi32>, 
                           %seq_lens: tensor<2xi32>,
                           %kv_cache: !llm.kvcache) 
                           -> tensor<2x8x16x64xf16> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<2x8x16x64xf16>
  // CHECK: %[[DIM1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<2x8x16x64xf16>
  // CHECK: %[[DIM2:.*]] = tensor.dim %arg0, %[[C2]] : tensor<2x8x16x64xf16>
  // CHECK: %[[DIM3:.*]] = tensor.dim %arg0, %[[C3]] : tensor<2x8x16x64xf16>
  // CHECK: %[[SCALE:.*]] = arith.constant 1.250000e-01 : f32
  // CHECK: %[[RESULT:.*]] = call @mlir_llm_paged_attention(%arg3, %arg0, %arg1, %arg2, %[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[SCALE]])
  // CHECK-SAME: : (!llm.kvcache, tensor<2x8x16x64xf16>, tensor<2x128xi32>, tensor<2xi32>, index, index, index, index, f32) -> tensor<2x8x16x64xf16>
  // CHECK: return %[[RESULT]]
  %output = llm.paged_attention(%query, %block_indices, %seq_lens, %kv_cache) {
    scale = 0.125 : f32
  } : (tensor<2x8x16x64xf16>, tensor<2x128xi32>, tensor<2xi32>, !llm.kvcache) 
      -> tensor<2x8x16x64xf16>
  return %output : tensor<2x8x16x64xf16>
} 