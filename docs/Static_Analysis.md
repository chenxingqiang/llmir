# LLMIR Static Analysis Documentation

This document describes the static analysis techniques used in LLMIR to enable compile-time optimizations for LLM inference workloads.

## 1. Overview

LLMIR performs several static analyses to enable optimizations that would otherwise require runtime information:

1. **Block Size Analysis**: Determines optimal KV cache block sizes
2. **Sequence Length Bounds Analysis**: Infers sequence length bounds for kernel selection
3. **Cache Sharing Analysis**: Detects cross-sequence KV cache sharing opportunities
4. **Quantization Safety Analysis**: Identifies operations safe for quantization
5. **Memory Requirement Analysis**: Computes memory bounds for allocation planning

---

## 2. Block Size Analysis

### 2.1 Problem Statement

The KV cache uses a paged memory layout where tokens are stored in fixed-size blocks. The block size significantly impacts performance:

- **Too small**: Increased overhead from block management, poor GPU utilization
- **Too large**: Memory waste due to fragmentation, reduced batch capacity

### 2.2 Analysis Algorithm

```
Algorithm: Block Size Optimization Analysis
═══════════════════════════════════════════════════════════════════════════════

INPUT:
  - IR module containing KV cache operations
  - Target hardware configuration H = (warp_size, cache_line_size, compute_capability)
  - Workload characteristics W = (expected_seq_lengths, batch_sizes)

OUTPUT:
  - Optimal block size B* for each KV cache in the module

ALGORITHM:

1. COLLECT SEQUENCE LENGTH INFORMATION
   ────────────────────────────────────
   For each llm.paged_kv_cache type T in module:
       // Extract static bounds
       max_seq_len ← T.maxSeqLen
       
       // Analyze operations using this cache
       For each op using cache of type T:
           If op is llm.append_kv:
               key_shape ← op.keys.type.shape
               seq_dim ← key_shape[1]
               If seq_dim is static:
                   observed_seq_lens.add(seq_dim)
           
           If op is llm.paged_attention:
               query_shape ← op.query.type.shape
               If query_shape[1] is static:
                   observed_seq_lens.add(query_shape[1])

2. COMPUTE FRAGMENTATION METRIC
   ────────────────────────────
   Function FragmentationScore(B, seq_lengths):
       total_waste ← 0
       For seq_len in seq_lengths:
           num_blocks ← ⌈seq_len / B⌉
           allocated ← num_blocks × B
           waste ← allocated - seq_len
           total_waste += waste / allocated
       Return total_waste / |seq_lengths|

3. COMPUTE GPU UTILIZATION METRIC
   ──────────────────────────────
   Function GPUUtilization(B, H):
       // Warp-level efficiency
       warp_util ← min(1.0, B / H.warp_size)
       
       // Memory coalescing efficiency
       elements_per_line ← H.cache_line_size / sizeof(element_type)
       coalescing ← 1.0 if B % elements_per_line == 0 else 0.7
       
       // Shared memory bank conflicts
       bank_conflicts ← 1.0 if B % 32 == 0 else 0.8
       
       Return warp_util × coalescing × bank_conflicts

4. OPTIMIZE BLOCK SIZE
   ────────────────────
   candidate_sizes ← [16, 32, 64, 128, 256]
   best_score ← -∞
   best_B ← 128  // Default
   
   For B in candidate_sizes:
       frag ← FragmentationScore(B, observed_seq_lens)
       util ← GPUUtilization(B, H)
       
       // Combined objective
       score ← (1.0 - frag) × util
       
       If score > best_score:
           best_score ← score
           best_B ← B
   
   Return best_B

═══════════════════════════════════════════════════════════════════════════════
```

### 2.3 Implementation

```cpp
// BlockSizeAnalysis.cpp

class BlockSizeAnalysis {
public:
    struct AnalysisResult {
        int64_t optimalBlockSize;
        double fragmentationScore;
        double gpuUtilization;
        double combinedScore;
    };
    
    AnalysisResult analyze(Operation* op, const HardwareConfig& hw) {
        // Collect sequence length information
        SmallVector<int64_t> seqLengths;
        collectSequenceLengths(op, seqLengths);
        
        // If no static information available, use defaults
        if (seqLengths.empty()) {
            return {128, 0.0, 0.0, 0.0};  // Default block size
        }
        
        // Evaluate candidate block sizes
        std::array<int64_t, 5> candidates = {16, 32, 64, 128, 256};
        AnalysisResult best = {128, 1.0, 0.0, -1.0};
        
        for (int64_t blockSize : candidates) {
            double frag = computeFragmentation(seqLengths, blockSize);
            double util = computeGPUUtilization(blockSize, hw);
            double score = (1.0 - frag) * util;
            
            if (score > best.combinedScore) {
                best = {blockSize, frag, util, score};
            }
        }
        
        return best;
    }
    
private:
    void collectSequenceLengths(Operation* op, SmallVector<int64_t>& lengths) {
        op->walk([&](Operation* inner) {
            if (auto appendOp = dyn_cast<AppendKVOp>(inner)) {
                auto keysType = appendOp.getKeys().getType().cast<ShapedType>();
                if (keysType.hasStaticShape()) {
                    lengths.push_back(keysType.getDimSize(1));
                }
            }
            if (auto attnOp = dyn_cast<PagedAttentionOp>(inner)) {
                auto queryType = attnOp.getQuery().getType().cast<ShapedType>();
                if (queryType.hasStaticShape()) {
                    lengths.push_back(queryType.getDimSize(1));
                }
            }
        });
    }
    
    double computeFragmentation(ArrayRef<int64_t> seqLengths, int64_t blockSize) {
        if (seqLengths.empty()) return 0.0;
        
        double totalWaste = 0.0;
        for (int64_t seqLen : seqLengths) {
            int64_t numBlocks = (seqLen + blockSize - 1) / blockSize;
            int64_t allocated = numBlocks * blockSize;
            totalWaste += static_cast<double>(allocated - seqLen) / allocated;
        }
        return totalWaste / seqLengths.size();
    }
    
    double computeGPUUtilization(int64_t blockSize, const HardwareConfig& hw) {
        double warpUtil = std::min(1.0, static_cast<double>(blockSize) / hw.warpSize);
        double coalescing = (blockSize * 2 % hw.cacheLineSize == 0) ? 1.0 : 0.7;
        double bankConflicts = (blockSize % 32 == 0) ? 1.0 : 0.8;
        return warpUtil * coalescing * bankConflicts;
    }
};
```

### 2.4 Workload-Specific Recommendations

| Workload Type | Typical Seq Lengths | Recommended Block Size |
|--------------|---------------------|----------------------|
| Chat/Dialogue | 100-2000 | 64 or 128 |
| Document QA | 2000-8000 | 128 or 256 |
| Code Generation | 500-4000 | 128 |
| Summarization | 1000-4000 | 128 |
| Real-time Inference | 10-200 | 32 or 64 |

---

## 3. Sequence Length Bounds Analysis

### 3.1 Purpose

Knowing sequence length bounds at compile time enables:
- **Kernel selection**: Choose FlashAttention vs standard attention
- **Memory planning**: Pre-allocate appropriate buffer sizes
- **Parallelization**: Determine optimal work distribution

### 3.2 Analysis Approach

```
Algorithm: Sequence Length Bounds Inference
═══════════════════════════════════════════════════════════════════════════════

INPUT:
  - Function containing attention operations
  
OUTPUT:
  - For each attention operation: (min_seq_len, max_seq_len, is_exact)

ANALYSIS:

1. DATAFLOW ANALYSIS
   ─────────────────
   For each llm.paged_attention op:
       query ← op.query
       
       // Trace query back to its definition
       query_def ← findDefinition(query)
       
       // Extract shape information
       If query_def has static shape:
           seq_len ← query_def.shape[1]
           Return (seq_len, seq_len, true)  // Exact bound
       
       // Check for shape inference attributes
       If query has "min_seq_len" attribute:
           min_len ← query.attr("min_seq_len")
           max_len ← query.attr("max_seq_len")
           Return (min_len, max_len, false)
       
       // Analyze KV cache type
       kv_cache ← op.kv_cache
       If kv_cache.type is PagedKVCacheType:
           max_len ← kv_cache.type.maxSeqLen
           Return (1, max_len, false)  // Conservative bound

2. INTERVAL ARITHMETIC
   ────────────────────
   For tensor operations:
       // Track sequence dimension through reshapes, slices
       If op is tensor.reshape:
           propagate bounds through reshape mapping
       If op is tensor.extract_slice:
           intersect bounds with slice specification
       If op is tensor.concat:
           union bounds of concatenated tensors

═══════════════════════════════════════════════════════════════════════════════
```

### 3.3 Kernel Selection Based on Bounds

```cpp
enum class AttentionKernel {
    STANDARD,       // seq_len <= 128
    FLASH_SMALL,    // 128 < seq_len <= 512
    FLASH_MEDIUM,   // 512 < seq_len <= 2048
    FLASH_LARGE,    // 2048 < seq_len <= 8192
    CHUNKED         // seq_len > 8192 or memory constrained
};

AttentionKernel selectKernel(int64_t minSeqLen, int64_t maxSeqLen, 
                             bool isExact, const TargetInfo& target) {
    // If exact bound, select precisely
    if (isExact) {
        if (maxSeqLen <= 128) return AttentionKernel::STANDARD;
        if (maxSeqLen <= 512) return AttentionKernel::FLASH_SMALL;
        if (maxSeqLen <= 2048) return AttentionKernel::FLASH_MEDIUM;
        if (maxSeqLen <= 8192) return AttentionKernel::FLASH_LARGE;
        return AttentionKernel::CHUNKED;
    }
    
    // For variable bounds, select based on expected common case
    // with fallback handling
    if (maxSeqLen <= 512) {
        // All cases fit in FLASH_SMALL
        return AttentionKernel::FLASH_SMALL;
    }
    
    // Generate multi-variant code with runtime dispatch
    // Compile multiple kernels and select at runtime
    return AttentionKernel::FLASH_MEDIUM;  // Default with runtime check
}
```

---

## 4. Cache Sharing Analysis

### 4.1 Problem Statement

In LLM serving, multiple requests may share common prompt prefixes (e.g., system prompts). Detecting these at compile time enables:
- **Memory efficiency**: Share KV cache blocks across sequences
- **Computation reuse**: Avoid redundant attention computation

### 4.2 Analysis Algorithm

```
Algorithm: Cross-Sequence Cache Sharing Detection
═══════════════════════════════════════════════════════════════════════════════

INPUT:
  - IR containing multiple sequences/batches
  - llm.append_kv operations

OUTPUT:
  - Sharing groups: sets of operations that can share KV cache blocks

ANALYSIS:

1. BUILD OPERATION GRAPH
   ─────────────────────
   For each llm.append_kv op:
       key_source ← traceKeySource(op.keys)
       value_source ← traceValueSource(op.values)
       op_info ← {op, key_source, value_source, seq_position}
       operations.add(op_info)

2. IDENTIFY SHARING CANDIDATES
   ───────────────────────────
   sharing_groups ← []
   
   For each pair (op1, op2) in operations:
       If op1.seq_position != op2.seq_position:
           continue  // Different positions can't share
       
       If canShare(op1.key_source, op2.key_source) AND
          canShare(op1.value_source, op2.value_source):
           // These operations compute identical KV pairs
           group ← findOrCreateGroup(op1, op2)
           sharing_groups.add(group)
   
   Return sharing_groups

3. SHARING CRITERIA
   ─────────────────
   Function canShare(source1, source2):
       // Same constant input
       If source1 and source2 are constant AND source1 == source2:
           Return true
       
       // Same computation from identical inputs
       If isIdenticalComputation(source1, source2):
           Return true
       
       // Marked as system prompt / shared prefix
       If hasAttribute(source1, "shared_prefix") AND
          hasAttribute(source2, "shared_prefix") AND
          getAttribute(source1, "prefix_id") == getAttribute(source2, "prefix_id"):
           Return true
       
       Return false

4. GENERATE SHARING ANNOTATIONS
   ────────────────────────────
   For each sharing_group:
       primary_op ← selectPrimaryOp(sharing_group)
       For op in sharing_group:
           If op != primary_op:
               op.setAttr("shares_with", primary_op.id)
               op.setAttr("enable_sharing", true)

═══════════════════════════════════════════════════════════════════════════════
```

### 4.3 Implementation

```cpp
class CacheSharingAnalysis {
public:
    struct SharingGroup {
        SmallVector<AppendKVOp> ops;
        int64_t sharedTokens;
        Value primaryCache;
    };
    
    SmallVector<SharingGroup> analyze(func::FuncOp func) {
        SmallVector<SharingGroup> groups;
        
        // Collect all append_kv operations
        SmallVector<AppendKVOp> appendOps;
        func.walk([&](AppendKVOp op) {
            appendOps.push_back(op);
        });
        
        // Build sharing groups
        DenseSet<Operation*> processed;
        for (auto op1 : appendOps) {
            if (processed.contains(op1))
                continue;
            
            SharingGroup group;
            group.ops.push_back(op1);
            
            for (auto op2 : appendOps) {
                if (op1 == op2 || processed.contains(op2))
                    continue;
                
                if (canShareKV(op1, op2)) {
                    group.ops.push_back(op2);
                    processed.insert(op2);
                }
            }
            
            if (group.ops.size() > 1) {
                group.sharedTokens = computeSharedTokens(group);
                groups.push_back(std::move(group));
            }
            processed.insert(op1);
        }
        
        return groups;
    }
    
private:
    bool canShareKV(AppendKVOp op1, AppendKVOp op2) {
        // Check if keys come from same source
        Value keys1 = op1.getKeys();
        Value keys2 = op2.getKeys();
        
        // Same defining operation
        if (keys1.getDefiningOp() == keys2.getDefiningOp())
            return true;
        
        // Both are constants with same value
        if (auto const1 = keys1.getDefiningOp<arith::ConstantOp>()) {
            if (auto const2 = keys2.getDefiningOp<arith::ConstantOp>()) {
                return const1.getValue() == const2.getValue();
            }
        }
        
        // Check for shared_prefix attribute
        if (op1->hasAttr("shared_prefix") && op2->hasAttr("shared_prefix")) {
            auto id1 = op1->getAttrOfType<StringAttr>("prefix_id");
            auto id2 = op2->getAttrOfType<StringAttr>("prefix_id");
            return id1 && id2 && id1.getValue() == id2.getValue();
        }
        
        return false;
    }
    
    int64_t computeSharedTokens(const SharingGroup& group) {
        if (group.ops.empty()) return 0;
        auto keysType = group.ops[0].getKeys().getType().cast<ShapedType>();
        if (!keysType.hasStaticShape()) return 0;
        return keysType.getDimSize(1);  // Sequence length dimension
    }
};
```

### 4.4 Sharing Optimization Pass

```cpp
struct CacheSharingOptimization : public OpRewritePattern<AppendKVOp> {
    using OpRewritePattern<AppendKVOp>::OpRewritePattern;
    
    LogicalResult matchAndRewrite(AppendKVOp op,
                                  PatternRewriter& rewriter) const override {
        // Check if this op can share with another
        auto sharesWithAttr = op->getAttrOfType<IntegerAttr>("shares_with");
        if (!sharesWithAttr)
            return failure();
        
        // Find the primary operation
        int64_t primaryId = sharesWithAttr.getInt();
        AppendKVOp primaryOp = findOpById(op->getParentOfType<func::FuncOp>(), 
                                          primaryId);
        if (!primaryOp)
            return failure();
        
        // Replace with lookup from shared cache
        auto lookupOp = rewriter.create<LookupKVOp>(
            op.getLoc(),
            primaryOp.getBlockIndices(),
            op.getSeqLens(),
            primaryOp.getUpdatedCache()
        );
        
        rewriter.replaceOp(op, {primaryOp.getUpdatedCache(), 
                                lookupOp.getBlockIndices()});
        return success();
    }
};
```

---

## 5. Quantization Safety Analysis

### 5.1 Problem Statement

Quantizing operations to lower precision (INT8/INT4) can cause accuracy degradation. Static analysis identifies which operations can be safely quantized.

### 5.2 Safety Criteria

| Operation Type | Quantization Safety | Reason |
|---------------|---------------------|--------|
| Linear (weights) | ✅ Safe for INT8/INT4 | Weights have bounded range |
| Linear (activations) | ✅ Safe for INT8 | Activation functions bound output |
| Attention QK^T | ⚠️ Partial (FP16 recommended) | Softmax sensitive to precision |
| Softmax | ❌ Unsafe | Exponential amplifies errors |
| LayerNorm | ⚠️ Partial (FP16 minimum) | Division sensitive |
| Residual Add | ⚠️ Depends on depth | Error accumulates |
| Embedding Lookup | ✅ Safe | Direct lookup, no computation |

### 5.3 Analysis Algorithm

```
Algorithm: Quantization Safety Analysis
═══════════════════════════════════════════════════════════════════════════════

INPUT:
  - IR module
  - Quantization configuration (target bits, error threshold)

OUTPUT:
  - For each operation: (can_quantize, recommended_bits, confidence)

ANALYSIS:

1. OPERATION CLASSIFICATION
   ─────────────────────────
   For each op in module:
       category ← classifyOperation(op)
       
       Switch category:
           Case LINEAR_WEIGHT:
               base_safety ← HIGH
               recommended_bits ← [4, 8]
           
           Case LINEAR_ACTIVATION:
               base_safety ← MEDIUM
               recommended_bits ← [8]
           
           Case ATTENTION_LOGITS:
               base_safety ← LOW
               recommended_bits ← [16]  // FP16
           
           Case SOFTMAX, LAYERNORM:
               base_safety ← NONE
               recommended_bits ← [16, 32]
           
           Case RESIDUAL:
               base_safety ← analyzeResidualSafety(op)

2. ERROR PROPAGATION ANALYSIS
   ───────────────────────────
   // Build dataflow graph
   dfg ← buildDataflowGraph(module)
   
   // Propagate quantization error bounds
   For op in topologicalOrder(dfg):
       input_errors ← [getErrorBound(input) for input in op.inputs]
       
       output_error ← propagateError(op, input_errors)
       setErrorBound(op.output, output_error)
       
       // Check if error exceeds threshold
       If output_error > config.error_threshold:
           markAsSensitive(op)

3. RESIDUAL DEPTH ANALYSIS
   ────────────────────────
   Function analyzeResidualSafety(residual_op):
       // Count number of residual connections to output
       depth ← countResidualDepth(residual_op)
       
       // Error accumulates with depth
       accumulated_error ← base_quant_error × sqrt(depth)
       
       If accumulated_error < threshold:
           Return HIGH  // Safe for INT8
       Elif accumulated_error < 2 × threshold:
           Return MEDIUM  // Use FP16
       Else:
           Return LOW  // Use FP32

4. GENERATE QUANTIZATION PLAN
   ───────────────────────────
   plan ← QuantizationPlan()
   
   For op in module:
       If base_safety[op] == HIGH AND not isSensitive(op):
           plan.add(op, recommended_bits[op])
       Elif base_safety[op] == MEDIUM:
           If not isSensitive(op):
               plan.add(op, 8)  // Conservative INT8
       Else:
           plan.keepFloat(op)
   
   Return plan

═══════════════════════════════════════════════════════════════════════════════
```

### 5.4 Implementation

```cpp
class QuantizationSafetyAnalysis {
public:
    enum class SafetyLevel {
        HIGH,    // Safe for INT4/INT8
        MEDIUM,  // Safe for INT8 only
        LOW,     // FP16 recommended
        NONE     // Must keep FP32
    };
    
    struct AnalysisResult {
        SafetyLevel level;
        int recommendedBits;
        double confidenceScore;
        std::string reason;
    };
    
    AnalysisResult analyze(Operation* op) {
        // Classify operation
        if (auto linear = dyn_cast<llm::LinearOp>(op)) {
            return analyzeLinear(linear);
        }
        if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
            return analyzeMatmul(matmul);
        }
        if (auto attention = dyn_cast<llm::AttentionOp>(op)) {
            return analyzeAttention(attention);
        }
        if (auto softmax = dyn_cast<llm::SoftmaxOp>(op)) {
            return {SafetyLevel::NONE, 32, 1.0, "Softmax requires high precision"};
        }
        
        // Default: conservative
        return {SafetyLevel::LOW, 16, 0.5, "Unknown operation type"};
    }
    
private:
    AnalysisResult analyzeLinear(llm::LinearOp op) {
        // Check if this is weight-only quantization
        Value weight = op.getWeight();
        if (isConstant(weight)) {
            // Weights can be safely quantized
            return {SafetyLevel::HIGH, 8, 0.95, "Static weight, safe for quantization"};
        }
        
        // Dynamic input - more conservative
        return {SafetyLevel::MEDIUM, 8, 0.8, "Dynamic input, use INT8"};
    }
    
    AnalysisResult analyzeAttention(llm::AttentionOp op) {
        // Attention logits are sensitive
        // But Q/K/V projections can be quantized
        return {SafetyLevel::LOW, 16, 0.9, 
                "Attention computation sensitive, use FP16"};
    }
    
    AnalysisResult analyzeMatmul(linalg::MatmulOp op) {
        // Check operand sources
        bool lhsConstant = isConstant(op.getInputs()[0]);
        bool rhsConstant = isConstant(op.getInputs()[1]);
        
        if (lhsConstant || rhsConstant) {
            return {SafetyLevel::HIGH, 8, 0.9, "One operand is constant"};
        }
        
        // Both dynamic - check downstream uses
        if (feedsIntoSoftmax(op)) {
            return {SafetyLevel::LOW, 16, 0.85, "Feeds into softmax"};
        }
        
        return {SafetyLevel::MEDIUM, 8, 0.75, "Dynamic matmul"};
    }
    
    bool isConstant(Value v) {
        return v.getDefiningOp<arith::ConstantOp>() != nullptr;
    }
    
    bool feedsIntoSoftmax(Operation* op) {
        for (auto user : op->getResult(0).getUsers()) {
            if (isa<llm::SoftmaxOp>(user))
                return true;
        }
        return false;
    }
};
```

### 5.5 Mixed-Precision Assignment

Based on the safety analysis, the compiler generates a mixed-precision plan:

```mlir
// Before: All FP16
func.func @transformer_block(%input: tensor<1x512x4096xf16>) -> tensor<1x512x4096xf16> {
    %q = llm.linear %input, %q_proj : ... -> tensor<1x512x4096xf16>
    %k = llm.linear %input, %k_proj : ... -> tensor<1x512x4096xf16>
    %v = llm.linear %input, %v_proj : ... -> tensor<1x512x4096xf16>
    %attn = llm.attention %q, %k, %v : ... -> tensor<1x512x4096xf16>
    ...
}

// After: Mixed precision based on analysis
func.func @transformer_block(%input: tensor<1x512x4096xf16>) -> tensor<1x512x4096xf16> {
    // Linear projections: INT8 quantized weights
    %q = llm.quantized_linear %input, %q_proj_i8, %q_scales : ... -> tensor<1x512x4096xf16>
    %k = llm.quantized_linear %input, %k_proj_i8, %k_scales : ... -> tensor<1x512x4096xf16>
    %v = llm.quantized_linear %input, %v_proj_i8, %v_scales : ... -> tensor<1x512x4096xf16>
    
    // Attention: Keep FP16 for numerical stability
    %attn = llm.attention %q, %k, %v : ... -> tensor<1x512x4096xf16>
    ...
}
```

---

## 6. Memory Requirement Analysis

### 6.1 Purpose

Compute memory bounds to enable:
- **Allocation planning**: Pre-allocate memory pools
- **Out-of-memory prevention**: Detect potential OOM before execution
- **Kernel selection**: Choose memory-efficient algorithms when constrained

### 6.2 Analysis Components

```
Memory Components in LLM Inference:
═══════════════════════════════════════════════════════════════════════════════

1. MODEL WEIGHTS
   M_weights = Σ (param_size × sizeof(dtype))
   
   For quantized models:
   M_weights_quant = Σ (param_size × bits / 8) + scale_overhead

2. KV CACHE
   M_kv = 2 × num_layers × num_heads × max_seq_len × head_dim × sizeof(dtype) × batch_size
   
   With paging:
   M_kv_paged = 2 × num_layers × num_heads × num_blocks × block_size × head_dim × sizeof(dtype)

3. ACTIVATION MEMORY
   M_act = max(activation_size) across all layers
   
   With activation checkpointing:
   M_act_ckpt = checkpoint_interval × activation_size_per_layer

4. TEMPORARY BUFFERS
   M_temp = attention_temp + matmul_temp + ...
   
   FlashAttention reduces this significantly:
   M_temp_flash = O(batch × heads × seq_len) vs O(batch × heads × seq_len²)

═══════════════════════════════════════════════════════════════════════════════
```

### 6.3 Static Memory Analysis

```cpp
class MemoryAnalysis {
public:
    struct MemoryEstimate {
        size_t weightsBytes;
        size_t kvCacheBytes;
        size_t activationBytes;
        size_t temporaryBytes;
        size_t totalPeakBytes;
    };
    
    MemoryEstimate analyze(ModuleOp module, const InferenceConfig& config) {
        MemoryEstimate estimate = {};
        
        // Analyze weights
        module.walk([&](arith::ConstantOp constOp) {
            if (auto tensorType = constOp.getType().dyn_cast<TensorType>()) {
                estimate.weightsBytes += computeTensorSize(tensorType);
            }
        });
        
        // Analyze KV cache
        module.walk([&](PagedKVCacheType cacheType) {
            size_t perSequence = 2 *  // K and V
                                 cacheType.getNumLayers() *
                                 cacheType.getNumHeads() *
                                 cacheType.getMaxSeqLen() *
                                 cacheType.getHeadDim() *
                                 getTypeSize(cacheType.getElementType());
            estimate.kvCacheBytes += perSequence * config.maxBatchSize;
        });
        
        // Analyze activation memory
        estimate.activationBytes = analyzeActivations(module, config);
        
        // Analyze temporary buffers
        estimate.temporaryBytes = analyzeTemporaries(module, config);
        
        // Peak = max concurrent memory
        estimate.totalPeakBytes = estimate.weightsBytes + 
                                  estimate.kvCacheBytes +
                                  estimate.activationBytes +
                                  estimate.temporaryBytes;
        
        return estimate;
    }
    
private:
    size_t analyzeActivations(ModuleOp module, const InferenceConfig& config) {
        size_t maxActivation = 0;
        
        module.walk([&](func::FuncOp func) {
            // Track live tensors through function
            DenseMap<Value, size_t> liveTensors;
            size_t currentMemory = 0;
            
            func.walk([&](Operation* op) {
                // Add outputs
                for (auto result : op->getResults()) {
                    if (auto tensorType = result.getType().dyn_cast<TensorType>()) {
                        size_t size = computeTensorSize(tensorType) * config.maxBatchSize;
                        liveTensors[result] = size;
                        currentMemory += size;
                    }
                }
                
                // Remove dead tensors
                for (auto& [value, size] : liveTensors) {
                    if (value.use_empty() || isLastUse(value, op)) {
                        currentMemory -= size;
                    }
                }
                
                maxActivation = std::max(maxActivation, currentMemory);
            });
        });
        
        return maxActivation;
    }
};
```

---

## 7. Analysis Pass Integration

### 7.1 Analysis Pass Infrastructure

```cpp
// AnalysisPasses.h

// Block Size Analysis Pass
class BlockSizeAnalysisPass 
    : public PassWrapper<BlockSizeAnalysisPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockSizeAnalysisPass)
    
    void runOnOperation() override {
        ModuleOp module = getOperation();
        BlockSizeAnalysis analysis;
        HardwareConfig hw = getHardwareConfig();
        
        module.walk([&](PagedKVCacheOp op) {
            auto result = analysis.analyze(op, hw);
            
            // Attach analysis results as attributes
            op->setAttr("optimal_block_size", 
                        IntegerAttr::get(IndexType::get(&getContext()), 
                                        result.optimalBlockSize));
            op->setAttr("fragmentation_score",
                        FloatAttr::get(Float64Type::get(&getContext()),
                                      result.fragmentationScore));
        });
    }
    
    StringRef getName() const override { return "BlockSizeAnalysis"; }
};

// Quantization Safety Analysis Pass
class QuantizationAnalysisPass
    : public PassWrapper<QuantizationAnalysisPass, OperationPass<ModuleOp>> {
public:
    void runOnOperation() override {
        ModuleOp module = getOperation();
        QuantizationSafetyAnalysis analysis;
        
        module.walk([&](Operation* op) {
            if (canBeQuantized(op)) {
                auto result = analysis.analyze(op);
                
                op->setAttr("quant_safety", 
                           StringAttr::get(&getContext(), 
                                          safetyLevelToString(result.level)));
                op->setAttr("quant_bits",
                           IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                           result.recommendedBits));
            }
        });
    }
};
```

### 7.2 Analysis Results Usage

```cpp
// Using analysis results in optimization passes

struct KVCacheOptimizationPass : public OpRewritePattern<PagedKVCacheOp> {
    LogicalResult matchAndRewrite(PagedKVCacheOp op,
                                  PatternRewriter& rewriter) const override {
        // Read analysis results
        auto blockSizeAttr = op->getAttrOfType<IntegerAttr>("optimal_block_size");
        if (!blockSizeAttr) {
            // Run analysis if not available
            BlockSizeAnalysis analysis;
            auto result = analysis.analyze(op, getHardwareConfig());
            blockSizeAttr = IntegerAttr::get(IndexType::get(op.getContext()),
                                            result.optimalBlockSize);
        }
        
        // Apply optimization
        int64_t optimalBlockSize = blockSizeAttr.getInt();
        if (op.getBlockSize() != optimalBlockSize) {
            // Create new cache type with optimized block size
            auto newCacheType = PagedKVCacheType::get(
                op.getContext(),
                op.getElementType(),
                op.getNumLayers(),
                op.getNumHeads(),
                op.getHeadDim(),
                optimalBlockSize,  // Updated
                op.getMaxSeqLen()
            );
            
            // Replace operation
            ...
        }
        
        return success();
    }
};
```

---

## 8. Debugging Analysis Results

### 8.1 Dumping Analysis Information

```bash
# Dump block size analysis
llmir-opt input.mlir \
    -llm-analyze-block-size \
    -llm-dump-analysis \
    2>&1 | grep "block_size"

# Output:
# llm.paged_kv_cache: optimal_block_size=128, fragmentation=0.12, gpu_util=0.95
```

### 8.2 Visualization

```python
from llmir.analysis import BlockSizeAnalysis, visualize

# Load compiled module
module = load_module("model.mlir")

# Run analysis
analysis = BlockSizeAnalysis(module)
results = analysis.run()

# Generate visualization
visualize.plot_fragmentation_vs_blocksize(results)
visualize.plot_gpu_utilization(results)
```

---

## 9. References

1. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
2. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention", NeurIPS 2022
3. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023
4. MLIR Analysis Infrastructure: https://mlir.llvm.org/docs/PassManagement/#analysis-management
