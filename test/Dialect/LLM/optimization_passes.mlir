// RUN: llmir-opt %s -llm-analyze-quantization -llm-analyze-block-size | FileCheck %s
// RUN: llmir-opt %s -llm-optimization-pipeline | FileCheck %s --check-prefix=OPTIMIZED

// This test file demonstrates LLMIR optimization passes and their effects.
// It covers:
// 1. Block size analysis and optimization
// 2. Quantization safety analysis
// 3. KV cache operation fusion
// 4. Pass pipeline orchestration

//===----------------------------------------------------------------------===//
// Test: Block Size Analysis
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_block_size_analysis
// CHECK: llm.optimal_block_size
func.func @test_block_size_analysis(
    %input: tensor<4x512x4096xf16>,
    // KV cache type: <element_type, num_layers, num_heads, head_dim, block_size, max_seq_len>
    %kv_cache: !llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>,
    %block_indices: tensor<4x128xi32>,
    %seq_lens: tensor<4xi32>
) -> tensor<4x512x4096xf16> {
    // Query projection
    %query = "llm.linear"(%input) : (tensor<4x512x4096xf16>) -> tensor<4x512x4096xf16>
    
    // Reshape for multi-head attention: [batch, seq, heads, head_dim]
    %query_mha = tensor.expand_shape %query [[0], [1], [2, 3]] 
        : tensor<4x512x4096xf16> into tensor<4x512x32x128xf16>
    
    // CHECK: optimal_block_size
    // CHECK: fragmentation_score
    %attn_out = llm.paged_attention %query_mha, %kv_cache, %block_indices, %seq_lens {
        num_heads = 32 : i32,
        head_dim = 128 : i32,
        scale = 0.0883883476 : f32
    } : (tensor<4x512x32x128xf16>, !llm.paged_kv_cache<f16, 32, 32, 128, 64, 8192>,
         tensor<4x128xi32>, tensor<4xi32>) -> tensor<4x512x32x128xf16>
    
    %result = tensor.collapse_shape %attn_out [[0], [1], [2, 3]]
        : tensor<4x512x32x128xf16> into tensor<4x512x4096xf16>
    
    return %result : tensor<4x512x4096xf16>
}

//===----------------------------------------------------------------------===//
// Test: Quantization Safety Analysis
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_quantization_analysis
func.func @test_quantization_analysis(
    %input: tensor<4x512x4096xf16>,
    %weight: tensor<4096x4096xf16>,
    %q_weight: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
    %scales: tensor<4096xf32>
) -> tensor<4x512x4096xf16> {
    // Linear with FP16 weights - should be marked as quantizable
    // CHECK: quant_safety = "high"
    %linear_out = "llm.linear"(%input) : (tensor<4x512x4096xf16>) -> tensor<4x512x4096xf16>
    
    // Quantized matmul - already quantized, keep as is
    // CHECK: quant_safety
    %qmm_out = llm.quantized_matmul %linear_out, %q_weight, %scales 
        : tensor<4x512x4096xf16>, !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>, 
          tensor<4096xf32> -> tensor<4x512x4096xf16>
    
    // Softmax - should be marked as not quantizable
    // CHECK: quant_safety = "none"
    %softmax_out = "llm.softmax"(%qmm_out) : (tensor<4x512x4096xf16>) -> tensor<4x512x4096xf16>
    
    // Layer norm - should recommend FP16
    // CHECK: quant_safety = "low"
    %norm_out = "llm.layer_norm"(%softmax_out) : (tensor<4x512x4096xf16>) -> tensor<4x512x4096xf16>
    
    return %norm_out : tensor<4x512x4096xf16>
}

//===----------------------------------------------------------------------===//
// Test: KV Cache Operation Fusion
//===----------------------------------------------------------------------===//

// This test demonstrates the append_kv + paged_attention fusion pattern.

// CHECK-LABEL: func.func @test_kv_cache_fusion
// OPTIMIZED-LABEL: func.func @test_kv_cache_fusion
func.func @test_kv_cache_fusion(
    %query: tensor<4x1x32x128xf16>,
    %key: tensor<4x1x32x128xf16>,
    %value: tensor<4x1x32x128xf16>,
    %kv_cache: !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>,
    %seq_ids: tensor<4xi32>,
    %seq_lens: tensor<4xi32>
) -> tensor<4x1x32x128xf16> {
    // Append new K, V to cache
    // In optimized version, this may be fused with attention
    %new_cache, %new_indices = llm.append_kv(%key, %value, %seq_ids, %kv_cache)
        : (tensor<4x1x32x128xf16>, tensor<4x1x32x128xf16>, tensor<4xi32>,
           !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>)
        -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>, tensor<4x1xi32>)
    
    // Compute attention using paged KV cache
    // OPTIMIZED: May show fused operation
    %attn_out = llm.paged_attention %query, %new_cache, %new_indices, %seq_lens {
        num_heads = 32 : i32,
        head_dim = 128 : i32,
        scale = 0.0883883476 : f32
    } : (tensor<4x1x32x128xf16>, !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>,
         tensor<4x1xi32>, tensor<4xi32>) -> tensor<4x1x32x128xf16>
    
    return %attn_out : tensor<4x1x32x128xf16>
}

//===----------------------------------------------------------------------===//
// Test: Cross-Sequence Cache Sharing
//===----------------------------------------------------------------------===//

// This test demonstrates detection of cache sharing opportunities
// when multiple sequences share a common prefix.

// CHECK-LABEL: func.func @test_cache_sharing
func.func @test_cache_sharing(
    %shared_prefix_k: tensor<1x256x32x128xf16>,  // Common system prompt
    %shared_prefix_v: tensor<1x256x32x128xf16>,
    %user_k1: tensor<1x50x32x128xf16>,
    %user_v1: tensor<1x50x32x128xf16>,
    %user_k2: tensor<1x60x32x128xf16>,
    %user_v2: tensor<1x60x32x128xf16>,
    %kv_cache: !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>,
    %seq_id1: tensor<1xi32>,
    %seq_id2: tensor<1xi32>
) -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>) {
    // First sequence: shared prefix + user content
    // CHECK: enable_sharing
    %cache1, %indices1 = llm.append_kv(%shared_prefix_k, %shared_prefix_v, %seq_id1, %kv_cache) {
        shared_prefix = true,
        prefix_id = "system_prompt_v1"
    } : (tensor<1x256x32x128xf16>, tensor<1x256x32x128xf16>, tensor<1xi32>,
         !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>)
        -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>, tensor<1x256xi32>)
    
    %cache1b, %indices1b = llm.append_kv(%user_k1, %user_v1, %seq_id1, %cache1)
        : (tensor<1x50x32x128xf16>, tensor<1x50x32x128xf16>, tensor<1xi32>,
           !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>)
        -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>, tensor<1x50xi32>)
    
    // Second sequence: same shared prefix + different user content
    // Analyzer should detect this can share blocks with first sequence
    // CHECK: enable_sharing
    %cache2, %indices2 = llm.append_kv(%shared_prefix_k, %shared_prefix_v, %seq_id2, %cache1b) {
        shared_prefix = true,
        prefix_id = "system_prompt_v1"
    } : (tensor<1x256x32x128xf16>, tensor<1x256x32x128xf16>, tensor<1xi32>,
         !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>)
        -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>, tensor<1x256xi32>)
    
    %cache2b, %indices2b = llm.append_kv(%user_k2, %user_v2, %seq_id2, %cache2)
        : (tensor<1x60x32x128xf16>, tensor<1x60x32x128xf16>, tensor<1xi32>,
           !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>)
        -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>, tensor<1x60xi32>)
    
    return %cache2b : !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>
}

//===----------------------------------------------------------------------===//
// Test: Full Transformer Layer
//===----------------------------------------------------------------------===//

// This test shows a complete transformer layer with all optimizations applied.

// CHECK-LABEL: func.func @transformer_layer
// OPTIMIZED-LABEL: func.func @transformer_layer
func.func @transformer_layer(
    %input: tensor<4x512x4096xf16>,
    %kv_cache: !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>,
    %block_indices: tensor<4x128xi32>,
    %seq_lens: tensor<4xi32>,
    %seq_ids: tensor<4xi32>,
    // Quantized weight matrices
    %q_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
    %k_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
    %v_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
    %o_proj: !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
    %qkvo_scales: tensor<4096xf32>,
    %ffn_up: !llm.quantized_tensor<i8, [4096, 11008], true, true, 0, -1, 8>,
    %ffn_down: !llm.quantized_tensor<i8, [11008, 4096], true, true, 0, -1, 8>,
    %ffn_scales_up: tensor<11008xf32>,
    %ffn_scales_down: tensor<4096xf32>
) -> (tensor<4x512x4096xf16>, !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>) {
    
    // === Pre-normalization ===
    %norm_input = "llm.rms_norm"(%input) : (tensor<4x512x4096xf16>) -> tensor<4x512x4096xf16>
    
    // === Self-Attention ===
    // Q, K, V projections with quantized weights
    %query_flat = llm.quantized_matmul %norm_input, %q_proj, %qkvo_scales
        : tensor<4x512x4096xf16>, !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
          tensor<4096xf32> -> tensor<4x512x4096xf16>
    
    %key_flat = llm.quantized_matmul %norm_input, %k_proj, %qkvo_scales
        : tensor<4x512x4096xf16>, !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
          tensor<4096xf32> -> tensor<4x512x4096xf16>
    
    %value_flat = llm.quantized_matmul %norm_input, %v_proj, %qkvo_scales
        : tensor<4x512x4096xf16>, !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
          tensor<4096xf32> -> tensor<4x512x4096xf16>
    
    // Reshape for multi-head attention
    %query = tensor.expand_shape %query_flat [[0], [1], [2, 3]]
        : tensor<4x512x4096xf16> into tensor<4x512x32x128xf16>
    %key = tensor.expand_shape %key_flat [[0], [1], [2, 3]]
        : tensor<4x512x4096xf16> into tensor<4x512x32x128xf16>
    %value = tensor.expand_shape %value_flat [[0], [1], [2, 3]]
        : tensor<4x512x4096xf16> into tensor<4x512x32x128xf16>
    
    // Update KV cache and compute attention
    %new_cache, %new_indices = llm.append_kv(%key, %value, %seq_ids, %kv_cache)
        : (tensor<4x512x32x128xf16>, tensor<4x512x32x128xf16>, tensor<4xi32>,
           !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>)
        -> (!llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>, tensor<4x512xi32>)
    
    %attn_out = llm.paged_attention %query, %new_cache, %new_indices, %seq_lens {
        num_heads = 32 : i32,
        head_dim = 128 : i32,
        scale = 0.0883883476 : f32
    } : (tensor<4x512x32x128xf16>, !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>,
         tensor<4x512xi32>, tensor<4xi32>) -> tensor<4x512x32x128xf16>
    
    // Reshape and output projection
    %attn_flat = tensor.collapse_shape %attn_out [[0], [1], [2, 3]]
        : tensor<4x512x32x128xf16> into tensor<4x512x4096xf16>
    
    %attn_proj = llm.quantized_matmul %attn_flat, %o_proj, %qkvo_scales
        : tensor<4x512x4096xf16>, !llm.quantized_tensor<i8, [4096, 4096], true, true, 0, -1, 8>,
          tensor<4096xf32> -> tensor<4x512x4096xf16>
    
    // First residual connection
    %attn_residual = arith.addf %input, %attn_proj : tensor<4x512x4096xf16>
    
    // === Feed-Forward Network ===
    %norm_ffn = "llm.rms_norm"(%attn_residual) : (tensor<4x512x4096xf16>) -> tensor<4x512x4096xf16>
    
    // Up projection (with SiLU activation baked in)
    %ffn_up_out = llm.quantized_matmul %norm_ffn, %ffn_up, %ffn_scales_up
        : tensor<4x512x4096xf16>, !llm.quantized_tensor<i8, [4096, 11008], true, true, 0, -1, 8>,
          tensor<11008xf32> -> tensor<4x512x11008xf16>
    
    %ffn_act = "llm.silu"(%ffn_up_out) : (tensor<4x512x11008xf16>) -> tensor<4x512x11008xf16>
    
    // Down projection
    %ffn_down_out = llm.quantized_matmul %ffn_act, %ffn_down, %ffn_scales_down
        : tensor<4x512x11008xf16>, !llm.quantized_tensor<i8, [11008, 4096], true, true, 0, -1, 8>,
          tensor<4096xf32> -> tensor<4x512x4096xf16>
    
    // Second residual connection
    %output = arith.addf %attn_residual, %ffn_down_out : tensor<4x512x4096xf16>
    
    return %output, %new_cache : tensor<4x512x4096xf16>, !llm.paged_kv_cache<f16, 1, 32, 128, 16, 8192>
}

//===----------------------------------------------------------------------===//
// Test: Tensor Parallelism Pattern
//===----------------------------------------------------------------------===//

// This test shows tensor-parallel sharding for multi-GPU execution.

// CHECK-LABEL: func.func @tensor_parallel_layer
func.func @tensor_parallel_layer(
    %input: tensor<4x512x4096xf16>,
    %weight_shard: tensor<4096x1024xf16>  // Sharded weight (1/4 of full)
) -> tensor<4x512x4096xf16> {
    // Sharded linear (column parallelism)
    %partial = llm.sharded_linear %input, %weight_shard {
        shard_dim = 1 : i64,
        num_shards = 4 : i64,
        shard_id = 0 : i64
    } : tensor<4x512x4096xf16>, tensor<4096x1024xf16> -> tensor<4x512x1024xf16>
    
    // All-gather to reconstruct full output
    %full = llm.all_gather %partial {
        dim = 2 : i64,
        group_size = 4 : i64
    } : tensor<4x512x1024xf16> -> tensor<4x512x4096xf16>
    
    return %full : tensor<4x512x4096xf16>
}
