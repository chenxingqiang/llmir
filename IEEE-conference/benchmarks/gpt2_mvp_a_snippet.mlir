module {
  func.func @kv_micro_pipeline(
      %cache: !llm.paged_kv_cache<f32, 1, 4, 32, 1024, 128>,
      %keys: tensor<1x8x4x32xf32>,
      %values: tensor<1x8x4x32xf32>,
      %seq_ids: tensor<1xi32>,
      %query: tensor<1x1x4x32xf32>
  ) -> tensor<1x1x4x32xf32> {
    %new_cache, %block_indices = llm.append_kv %cache, %keys, %values, %seq_ids {
      block_size = 32 : i32,
      max_seq_len = 128 : i32
    } : (!llm.paged_kv_cache<f32, 1, 4, 32, 1024, 128>, tensor<1x8x4x32xf32>, tensor<1x8x4x32xf32>, tensor<1xi32>)
        -> (!llm.paged_kv_cache<f32, 1, 4, 32, 1024, 128>, tensor<1x1xi32>)
    %seq_lens = arith.constant dense<8> : tensor<1xi32>
    %k, %v = llm.lookup_kv %new_cache, %block_indices, %seq_lens {
      num_heads = 4 : i32,
      head_dim = 32 : i32
    } : (!llm.paged_kv_cache<f32, 1, 4, 32, 1024, 128>, tensor<1x1xi32>, tensor<1xi32>)
        -> (tensor<1x8x4x32xf32>, tensor<1x8x4x32xf32>)
    %out = llm.paged_attention %query, %new_cache, %block_indices, %seq_lens {
      num_heads = 4 : i32,
      head_dim = 32 : i32,
      scale = 0.17677670 : f32
    } : (tensor<1x1x4x32xf32>, !llm.paged_kv_cache<f32, 1, 4, 32, 1024, 128>, tensor<1x1xi32>, tensor<1xi32>)
        -> tensor<1x1x4x32xf32>
    return %out : tensor<1x1x4x32xf32>
  }
}
