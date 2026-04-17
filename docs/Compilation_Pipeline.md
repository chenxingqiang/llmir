# LLMIR Compilation Pipeline

This document describes the end-to-end compilation pipeline of LLMIR, from model import to optimized code generation.

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLMIR Compilation Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Stage 1   │ -> │   Stage 2   │ -> │   Stage 3   │ -> │   Stage 4   │   │
│  │   Model     │    │  High-Level │    │  Lowering & │    │   Runtime   │   │
│  │   Import    │    │ Optimization│    │  CodeGen    │    │ Integration │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│        │                  │                  │                  │            │
│        v                  v                  v                  v            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │ PyTorch/    │    │ LLM Dialect │    │ LLVM/NVVM   │    │ Executable  │   │
│  │ ONNX/HF     │    │ Passes      │    │ GPU Dialect │    │ + Runtime   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Stage 1: Model Import

### 2.1 Supported Input Formats

| Format | Import Mechanism | Use Case |
|--------|-----------------|----------|
| PyTorch | Torch-MLIR | Most common, direct model loading |
| ONNX | ONNX-MLIR | Cross-framework compatibility |
| HuggingFace | Custom importer | Direct HF model support |
| SafeTensors | Weight loading | Efficient weight serialization |

### 2.2 PyTorch Import Flow

```python
# Example: Import a HuggingFace model
import torch
from transformers import AutoModelForCausalLM
from llmir.importers import PyTorchImporter

# Load PyTorch model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Create sample input for tracing
sample_input = {
    "input_ids": torch.randint(0, 32000, (1, 512)),
    "attention_mask": torch.ones(1, 512)
}

# Import to LLMIR
importer = PyTorchImporter()
llmir_module = importer.import_model(
    model,
    sample_inputs=sample_input,
    target_dialect="llm"
)
```

### 2.3 Import Transformations

During import, the following transformations occur:

#### 2.3.1 Attention Pattern Recognition

The importer identifies attention patterns in the computation graph:

```python
# PyTorch attention pattern:
def attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

Becomes MLIR:

```mlir
%output = llm.attention %query, %key, %value, %mask { scale = 0.125 : f32, causal = true }
    : tensor<1x512x32x128xf16>, tensor<1x512x32x128xf16>, tensor<1x512x32x128xf16>, 
      tensor<1x1x512x512xi1> -> tensor<1x512x32x128xf16>
```

#### 2.3.2 KV Cache Insertion

For autoregressive models, the importer inserts KV cache operations:

```mlir
// Original: recomputes full attention every step
func.func @generate_step(%input: tensor<1x1x4096xf16>, ...) {
    %k = llm.linear %input, %k_proj : ...
    %v = llm.linear %input, %v_proj : ...
    %attn = llm.attention %q, %k, %v : ...
}

// After import: uses paged KV cache
func.func @generate_step(%input: tensor<1x1x4096xf16>, 
                         %kv_cache: !llm.paged_kv_cache<f16, 32, 32, 128, 16, 8192>, 
                         %block_indices: tensor<1x128xi32>,
                         %seq_lens: tensor<1xi32>) {
    %k = llm.linear %input, %k_proj : ...
    %v = llm.linear %input, %v_proj : ...
    %new_cache, %new_indices = llm.append_kv(%k, %v, %seq_ids, %kv_cache) : ...
    %attn = llm.paged_attention %q, %new_cache, %new_indices, %seq_lens : ...
}
```

#### 2.3.3 Weight Quantization Detection

Pre-quantized models (GPTQ, AWQ) are converted to quantized tensor types:

```mlir
// GPTQ-quantized weight becomes:
%q_weight = arith.constant dense<...> : !llm.quantized_tensor<i4, [4096, 4096], true, false, -1, 128, 4>
%scales = arith.constant dense<...> : tensor<32x4096xf16>
%zeros = arith.constant dense<...> : tensor<32x4096xi4>
```

### 2.4 Model Architecture Support

| Model Family | Import Status | Notes |
|--------------|--------------|-------|
| LLaMA/LLaMA-2 | ✅ Full | Standard transformer |
| Mistral | ✅ Full | Sliding window attention |
| Phi-3 | ✅ Full | Block sparse attention |
| Qwen-2 | ✅ Full | Standard transformer |
| DeepSeek-V2 | ✅ Full | MoE architecture |
| GPT-NeoX | ✅ Full | Rotary embeddings |
| Falcon | ⚠️ Partial | Multi-query attention |

---

## 3. Stage 2: High-Level Optimization

### 3.1 Pass Orchestration

The optimization pipeline is organized into phases:

```cpp
// Pass pipeline definition
void buildLLMOptimizationPipeline(mlir::OpPassManager &pm) {
    // Phase 1: Canonicalization and simplification
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    
    // Phase 2: LLM-specific optimizations
    pm.addNestedPass<func::FuncOp>(llm::createKVCacheOptimizationPass());
    pm.addNestedPass<func::FuncOp>(llm::createAttentionFusionPass());
    pm.addNestedPass<func::FuncOp>(llm::createQuantizationOptimizationPass());
    
    // Phase 3: Parallelization (if multi-GPU)
    pm.addPass(llm::createTensorParallelismPass());
    pm.addPass(llm::createPipelineParallelismPass());
    pm.addPass(llm::createCommunicationOptimizationPass());
    
    // Phase 4: Pre-lowering cleanup
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
}
```

### 3.2 KV Cache Optimization Pass

**Algorithm: Block Size Optimization**

```
Algorithm 1: Block Size Optimization
─────────────────────────────────────────────────────────────────────
Input: KV cache operations, target hardware specs
Output: Optimized block size attribute

1. function OptimizeBlockSize(kv_ops, hardware):
2.     seq_lengths ← CollectSequenceLengthBounds(kv_ops)
3.     
4.     // Candidate block sizes (powers of 2)
5.     candidates ← [16, 32, 64, 128, 256]
6.     
7.     best_score ← -∞
8.     best_block_size ← 128  // Default
9.     
10.    for block_size in candidates:
11.        // Compute memory efficiency
12.        avg_waste ← ComputeAverageFragmentation(seq_lengths, block_size)
13.        
14.        // Compute compute efficiency
15.        util ← ComputeGPUUtilization(block_size, hardware.warp_size)
16.        
17.        // Memory alignment bonus
18.        align_bonus ← 1.0 if block_size % hardware.cache_line == 0 else 0.8
19.        
20.        // Combined score
21.        score ← (1.0 - avg_waste) × util × align_bonus
22.        
23.        if score > best_score:
24.            best_score ← score
25.            best_block_size ← block_size
26.    
27.    return best_block_size
─────────────────────────────────────────────────────────────────────
```

**Implementation:**

```cpp
// KVCacheOptimization.cpp
int64_t computeOptimalBlockSize(ArrayRef<int64_t> seqLengths, 
                                 const HardwareConfig& hw) {
    SmallVector<int64_t> candidates = {16, 32, 64, 128, 256};
    double bestScore = -1.0;
    int64_t bestBlockSize = 128;
    
    for (int64_t blockSize : candidates) {
        // Memory fragmentation: avg wasted space per sequence
        double totalWaste = 0.0;
        for (int64_t seqLen : seqLengths) {
            int64_t numBlocks = (seqLen + blockSize - 1) / blockSize;
            int64_t waste = numBlocks * blockSize - seqLen;
            totalWaste += static_cast<double>(waste) / (numBlocks * blockSize);
        }
        double avgWaste = totalWaste / seqLengths.size();
        
        // GPU warp utilization
        double util = std::min(1.0, static_cast<double>(blockSize) / hw.warpSize);
        
        // Cache line alignment
        double alignBonus = (blockSize * sizeof(float16_t) % hw.cacheLineSize == 0) 
                           ? 1.0 : 0.8;
        
        double score = (1.0 - avgWaste) * util * alignBonus;
        if (score > bestScore) {
            bestScore = score;
            bestBlockSize = blockSize;
        }
    }
    return bestBlockSize;
}
```

### 3.3 Attention Fusion Pass

Fuses patterns for more efficient computation:

**Pattern 1: AppendKV + PagedAttention Fusion**

```mlir
// Before fusion:
%cache2, %indices = llm.append_kv(%k, %v, %seq_ids, %cache1)
%output = llm.paged_attention %q, %cache2, %indices, %seq_lens { ... }

// After fusion:
%output, %cache2 = llm.fused_decode_attention %q, %k, %v, %seq_ids, %cache1, %seq_lens { ... }
```

**Pattern 2: Multi-Head Attention Unfusion (for FlashAttention)**

```mlir
// When FlashAttention kernel is available, unfuse to expose fusion opportunity
%output = llm.attention %q, %k, %v { scale = 0.125, causal = true }

// Lowered with FlashAttention:
%output = llm.flash_attention %q, %k, %v { 
    scale = 0.125, 
    causal = true,
    block_m = 128,
    block_n = 128
}
```

### 3.4 Quantization Optimization Pass

**Algorithm: Quantization Safety Analysis**

```
Algorithm 2: Quantization Safety Analysis
─────────────────────────────────────────────────────────────────────
Input: Operation graph, quantization config
Output: Safe quantization plan

1. function AnalyzeQuantizationSafety(ops, config):
2.     safe_ops ← {}
3.     sensitive_ops ← {}
4.     
5.     for op in ops:
6.         sensitivity ← ComputeSensitivity(op)
7.         
8.         if op.type == "linear" and sensitivity < config.threshold:
9.             safe_ops.add(op, config.default_bits)  // Usually INT8/INT4
10.        
11.        else if op.type == "attention.logits":
12.            sensitive_ops.add(op)  // Keep FP16/FP32
13.        
14.        else if op.type == "normalization":
15.            sensitive_ops.add(op)  // Keep FP16
16.        
17.        else if op.type == "residual_add":
18.            // Analyze accumulated error
19.            error ← EstimateAccumulatedError(op)
20.            if error < config.error_threshold:
21.                safe_ops.add(op, 16)  // FP16 accumulation
22.            else:
23.                sensitive_ops.add(op)  // FP32 accumulation
24.    
25.    return QuantizationPlan(safe_ops, sensitive_ops)
─────────────────────────────────────────────────────────────────────
```

### 3.5 Parallelization Passes

#### Tensor Parallelism Pass

Distributes model across GPUs by sharding weight matrices:

```mlir
// Before: Single-GPU linear layer
%output = llm.linear %input, %weight : tensor<b×4096xf16>, tensor<4096×16384xf16> 
    -> tensor<b×16384xf16>

// After: 4-GPU tensor parallel
%partial = llm.sharded_linear %input, %weight_shard { 
    shard_dim = 1, num_shards = 4, shard_id = %rank 
} : tensor<b×4096xf16>, tensor<4096×4096xf16> -> tensor<b×4096xf16>

%output = llm.all_gather %partial { dim = 1, group_size = 4 } 
    : tensor<b×4096xf16> -> tensor<b×16384xf16>
```

#### Pipeline Parallelism Pass

Distributes transformer layers across stages:

```mlir
// Hybrid parallelism: 8 GPUs = 2 pipeline stages × 4 tensor parallel
// Stage 0: Layers 0-15 on GPUs [0,1,2,3]
// Stage 1: Layers 16-31 on GPUs [4,5,6,7]

func.func @pipeline_stage_0(%input, %kv_caches) {
    // Layers 0-15 with tensor parallelism across GPUs 0-3
    %out = llm.transformer_layers %input, %kv_caches { 
        start_layer = 0, end_layer = 15, tp_size = 4 
    }
    // Send to next stage
    llm.pipeline_send %out { dst_stage = 1 }
}

func.func @pipeline_stage_1(%kv_caches) {
    // Receive from previous stage
    %input = llm.pipeline_recv { src_stage = 0 }
    // Layers 16-31 with tensor parallelism across GPUs 4-7
    %out = llm.transformer_layers %input, %kv_caches { 
        start_layer = 16, end_layer = 31, tp_size = 4 
    }
    return %out
}
```

---

## 4. Stage 3: Lowering and Code Generation

### 4.1 Lowering Hierarchy

```
LLM Dialect
    │
    ▼
┌─────────────────┐
│  LLM Lowering   │  llm.attention → linalg.batch_matmul + softmax
│     Pass        │  llm.paged_attention → runtime calls OR specialized kernels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linalg/Tensor   │  linalg.matmul, tensor.extract, arith ops
│   Dialects      │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌───────┐  ┌───────┐
│  GPU  │  │  CPU  │
│Dialect│  │Dialect│
└───┬───┘  └───┬───┘
    │          │
    ▼          ▼
┌───────┐  ┌───────┐
│ NVVM/ │  │ LLVM  │
│  PTX  │  │  IR   │
└───────┘  └───────┘
```

### 4.2 GPU Code Generation

#### Kernel Selection Strategy

The compiler selects kernels based on workload characteristics:

```cpp
enum KernelVariant {
    FLASH_PAGED,      // FlashAttention-style, seq_len > 512
    STANDARD_PAGED,   // Standard attention, seq_len <= 512  
    CHUNKED_PAGED,    // Memory-constrained, large batch
    DECODE_OPTIMIZED  // Single-token decode, latency-critical
};

KernelVariant selectKernel(const PagedAttentionOp& op, 
                           const TargetInfo& target) {
    auto queryShape = op.getQuery().getType().getShape();
    int64_t seqLen = queryShape[1];
    int64_t batchSize = queryShape[0];
    
    // Decode phase: single token
    if (seqLen == 1) {
        return DECODE_OPTIMIZED;
    }
    
    // Prefill phase: choose based on sequence length
    if (seqLen > 512 && target.hasFlashAttention()) {
        return FLASH_PAGED;
    }
    
    // Memory-constrained scenarios
    size_t requiredMem = estimateMemory(op);
    if (requiredMem > target.availableMemory * 0.8) {
        return CHUNKED_PAGED;
    }
    
    return STANDARD_PAGED;
}
```

#### FlashAttention Implementation

```mlir
// High-level llm.paged_attention lowers to FlashAttention kernel:
llm.paged_attention %query, %kv_cache, %indices, %seq_lens {
    num_heads = 32, head_dim = 128, scale = 0.0884
}

// Lowers to GPU kernel launch:
gpu.launch_func @flash_paged_attention_kernel 
    blocks in (%batch_size, %num_heads, 1)
    threads in (128, 1, 1)
    args(%query_ptr, %kv_cache_ptr, %indices_ptr, %seq_lens_ptr, %output_ptr,
         %scale, %block_size, %max_seq_len)
```

The FlashAttention kernel implements:

```cuda
// Pseudocode for flash_paged_attention_kernel
__global__ void flash_paged_attention_kernel(
    float16_t* query,      // [batch, seq_q, heads, head_dim]
    KVCache* kv_cache,     // Paged KV cache
    int32_t* block_indices,// [batch, max_blocks]
    int32_t* seq_lens,     // [batch]
    float16_t* output,     // [batch, seq_q, heads, head_dim]
    float scale,
    int block_size,
    int max_seq_len
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_len = seq_lens[batch_idx];
    
    // Tiled computation
    __shared__ float16_t q_tile[BLOCK_M][HEAD_DIM];
    __shared__ float16_t k_tile[BLOCK_N][HEAD_DIM];
    __shared__ float16_t v_tile[BLOCK_N][HEAD_DIM];
    
    // Load query tile to shared memory
    load_query_tile(q_tile, query, batch_idx, head_idx);
    
    float m_prev = -INFINITY;  // Running max
    float l_prev = 0.0f;       // Running sum
    float16_t o_acc[HEAD_DIM] = {0};  // Accumulator
    
    // Iterate over KV blocks
    for (int block_idx = 0; block_idx < (seq_len + block_size - 1) / block_size; block_idx++) {
        int kv_block = block_indices[batch_idx * max_blocks + block_idx];
        
        // Load K, V from paged cache
        load_kv_block(k_tile, v_tile, kv_cache, kv_block, head_idx);
        
        // Compute attention scores for this block
        float scores[BLOCK_N];
        compute_qk(q_tile, k_tile, scores, scale);
        
        // Online softmax update
        float m_new = max(m_prev, max(scores));
        float l_new = exp(m_prev - m_new) * l_prev + sum(exp(scores - m_new));
        
        // Update output accumulator
        update_output(o_acc, v_tile, scores, m_prev, m_new, l_prev, l_new);
        
        m_prev = m_new;
        l_prev = l_new;
    }
    
    // Final normalization and store
    normalize_and_store(output, o_acc, l_prev, batch_idx, head_idx);
}
```

### 4.3 CPU Code Generation

For CPU targets, operations lower to optimized LLVM IR with vectorization:

```mlir
// llm.quantized_matmul on CPU lowers to:
scf.parallel (%i, %j) = (%c0, %c0) to (%M, %N) step (%c8, %c8) {
    // Load 8x8 block with AVX-512 intrinsics
    %a = vector.load %lhs[%i, %k] : memref<MxKxf16>, vector<8xf16>
    %b_q = vector.load %rhs_quantized[%k, %j] : memref<KxNxi8>, vector<8xi8>
    %s = vector.load %scales[%j] : memref<Nxf32>, vector<8xf32>
    
    // Dequantize and compute
    %b = arith.sitofp %b_q : vector<8xi8> to vector<8xf16>
    %b_scaled = arith.mulf %b, %s : vector<8xf16>
    %c = vector.fma %a, %b_scaled, %acc : vector<8xf16>
    
    vector.store %c, %output[%i, %j] : memref<MxNxf16>, vector<8xf16>
}
```

---

## 5. Stage 4: Runtime Integration

### 5.1 Runtime Library Interface

The generated code interfaces with the LLMIR runtime library:

```cpp
// Runtime function declarations (generated by lowering)
extern "C" {
    // KV Cache management
    void* mlir_llm_create_paged_kv_cache(
        int64_t num_layers, int64_t num_heads, int64_t head_dim,
        int64_t block_size, int64_t max_seq_len, bool use_gpu);
    
    void mlir_llm_append_kv(
        void* kv_cache, void* keys, void* values, 
        int32_t* seq_ids, int64_t batch_size, int64_t seq_len,
        int32_t* out_block_indices);
    
    void mlir_llm_lookup_kv(
        void* kv_cache, int32_t* block_indices, int32_t* seq_lens,
        int64_t batch_size, void* out_keys, void* out_values);
    
    void mlir_llm_paged_attention(
        void* kv_cache, void* query, int32_t* block_indices,
        int32_t* seq_lens, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim, float scale,
        void* output);
    
    // Memory management
    void* mlir_llm_allocate_tensor(int64_t* shape, int64_t rank, int dtype);
    void mlir_llm_deallocate_tensor(void* tensor);
    
    // Collective operations
    void mlir_llm_all_gather(void* input, void* output, 
                             int64_t dim, int64_t group_size);
    void mlir_llm_reduce_scatter(void* input, void* output,
                                  int64_t dim, int64_t group_size, int reduce_op);
}
```

### 5.2 Runtime Memory Management

The runtime manages memory allocation and scheduling:

```cpp
class LLMRuntime {
public:
    // Create execution context
    static std::unique_ptr<LLMRuntime> create(const RuntimeConfig& config);
    
    // Load compiled model
    void loadModule(const std::string& path);
    
    // Execute inference
    Tensor generate(const Tensor& input_ids, const GenerationConfig& config);
    
    // Memory management
    void setMemoryLimit(size_t bytes);
    void enableMemoryPressureHandling(bool enable);
    
    // Continuous batching
    void addRequest(const Request& request);
    std::vector<Response> step();
    
private:
    std::unique_ptr<PagedKVCacheManager> kv_manager_;
    std::unique_ptr<BatchScheduler> scheduler_;
    std::unique_ptr<ExecutionEngine> engine_;
};
```

### 5.3 Compile-Time vs Runtime Responsibilities

| Aspect | Compile-Time | Runtime |
|--------|--------------|---------|
| Block Size | Optimal size selection | Dynamic block allocation |
| Memory Layout | Cache-friendly arrangement | Actual memory allocation |
| Kernel Selection | Variant selection based on bounds | Parameter binding |
| Sharing Detection | Cross-sequence sharing analysis | Sharing activation |
| Quantization | Safe operation identification | Dequantization execution |
| Parallelism | Partitioning and placement | Communication |

---

## 6. Using the Compiler

### 6.1 Command-Line Interface

```bash
# Full compilation pipeline
llmir-opt input.mlir \
    -llm-import-pytorch \
    -llm-optimize-kv-cache \
    -llm-attention-fusion \
    -llm-quantization-optimization \
    -llm-tensor-parallelism=num_gpus=4 \
    -llm-lower-to-gpu \
    -convert-gpu-to-nvvm \
    -o output.mlir

# Generate executable
llmir-translate output.mlir -emit-cuda -o output.cu
nvcc output.cu -o inference -lcudart -lcublas
```

### 6.2 Python API

```python
from llmir import Compiler, OptimizationConfig

# Configure compiler
config = OptimizationConfig(
    target="cuda",
    num_gpus=4,
    tensor_parallel=True,
    quantization="int8",
    block_size="auto",  # Automatic optimization
    enable_flash_attention=True
)

# Compile model
compiler = Compiler(config)
compiled_module = compiler.compile("model.onnx")

# Execute
runtime = compiled_module.create_runtime()
output = runtime.generate(input_ids, max_new_tokens=100)
```

### 6.3 Integration with vLLM

```python
from vllm import LLM, SamplingParams
from llmir.integration.vllm import LLMIRBackend

# Use LLMIR as backend for vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    backend=LLMIRBackend(
        optimization_level=3,
        quantization="int8",
        enable_flash_attention=True
    )
)

# Standard vLLM interface
outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
```

---

## 7. Performance Considerations

### 7.1 Optimization Impact

| Optimization | Throughput Gain | Memory Reduction |
|-------------|-----------------|------------------|
| PagedAttention | 1.5× | 60-70% |
| FlashAttention | 2-3× | 50% |
| INT8 Quantization | 1.2× | 50% |
| INT4 Quantization | 1.5× | 75% |
| Tensor Parallelism (4 GPU) | 3.5× | - |
| Continuous Batching | 2× | - |

### 7.2 Tuning Recommendations

1. **Block Size**: Use auto-optimization or profile with representative workloads
2. **Batch Size**: Larger batches improve throughput but increase latency
3. **Quantization**: INT8 for most operations, FP16 for attention scores
4. **Parallelism**: Match tensor parallel degree to memory constraints

---

## 8. Debugging and Profiling

### 8.1 IR Dumps

```bash
# Dump IR after each pass
llmir-opt input.mlir \
    -llm-optimize-kv-cache -print-ir-after=llm-optimize-kv-cache \
    -llm-attention-fusion -print-ir-after=llm-attention-fusion \
    2>&1 | tee optimization_trace.txt
```

### 8.2 Performance Profiling

```python
from llmir.profiling import Profiler

profiler = Profiler(compiled_module)
with profiler.profile():
    output = runtime.generate(input_ids)

print(profiler.summary())
# Output:
# Operation          Time (ms)   Memory (MB)   Calls
# llm.paged_attention   45.2        2048         32
# llm.quantized_matmul  23.1         512         64
# llm.append_kv          5.3         128         32
```

---

## 9. References

1. MLIR: Multi-Level Intermediate Representation Compiler Infrastructure
2. Torch-MLIR: Bridging PyTorch and MLIR
3. vLLM: Efficient Memory Management for Large Language Model Serving
4. FlashAttention: Fast and Memory-Efficient Exact Attention
5. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
