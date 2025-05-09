// Example showing the usage of the LLM dialect attention operations

// Define a basic attention computation
func.func @attention_example(%query: tensor<2x16x64xf16>, %key: tensor<2x16x64xf16>, %value: tensor<2x16x64xf16>) -> tensor<2x16x64xf16> {
  %result = llm.attention %query, %key, %value { scale = 0.125 : f32, causal = true } : 
    tensor<2x16x64xf16>, tensor<2x16x64xf16>, tensor<2x16x64xf16> -> tensor<2x16x64xf16>
  return %result : tensor<2x16x64xf16>
}

// Example with paged attention using KV cache
func.func @paged_attention_example(%query: tensor<1x1x16x64xf16>, %kv_cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, 
                              %block_idxs: tensor<1xi32>, %seq_lens: tensor<1xi32>) -> tensor<1x1x16x64xf16> {
  %result = llm.paged_attention %query, %kv_cache, %block_idxs, %seq_lens { scale = 0.125 : f32 } : 
    tensor<1x1x16x64xf16>, !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, tensor<1xi32>, tensor<1xi32> -> tensor<1x1x16x64xf16>
  return %result : tensor<1x1x16x64xf16>
}

// Example of KV cache management
func.func @append_kv_example(%kv_cache: !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>,
                        %key: tensor<1x1x16x64xf16>, %value: tensor<1x1x16x64xf16>,
                        %seq_ids: tensor<1xi32>) -> (!llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, tensor<1xi32>) {
  %updated_cache, %block_idxs = llm.append_kv %kv_cache, %key, %value, %seq_ids : 
    !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, tensor<1x1x16x64xf16>, tensor<1x1x16x64xf16>, tensor<1xi32> -> 
    !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, tensor<1xi32>
  return %updated_cache, %block_idxs : !llm.paged_kv_cache<f16, 12, 16, 64, 16, 2048>, tensor<1xi32>
}

// Example of quantization
func.func @quantize_example(%input: tensor<16x1024xf16>, %scales: tensor<1024xf32>) -> !llm.quantized_tensor<i8, [16, 1024], true, true, 1, 128, 8> {
  %result = llm.quantize %input, %scales { bits = 8 : i32, symmetric = true, axis = 1 : i64, group_size = 128 : i64 } : 
    tensor<16x1024xf16>, tensor<1024xf32> -> !llm.quantized_tensor<i8, [16, 1024], true, true, 1, 128, 8>
  return %result : !llm.quantized_tensor<i8, [16, 1024], true, true, 1, 128, 8>
}

// Example of tensor parallelism
func.func @sharded_linear_example(%input: tensor<16x1024xf16>, %weight: tensor<1024x1024xf16>, %bias: tensor<1024xf16>) -> tensor<16x1024xf16> {
  %result = llm.sharded_linear %input, %weight, %bias { shardDim = 1 : i64, numShards = 8 : i64, shardId = 0 : i64 } : 
    tensor<16x1024xf16>, tensor<1024x1024xf16>, tensor<1024xf16> -> tensor<16x1024xf16>
  return %result : tensor<16x1024xf16>
}

// Example of gather operation for tensor parallelism
func.func @all_gather_example(%shard: tensor<16x128xf16>) -> tensor<16x1024xf16> {
  %result = llm.all_gather %shard { dim = 1 : i64, groupSize = 8 : i64 } : tensor<16x128xf16> -> tensor<16x1024xf16>
  return %result : tensor<16x1024xf16>
} 