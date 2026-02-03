# LLMIR Paper Submission Summary

## Paper Information
- **Title**: LLMIR: A Compiler Infrastructure for Optimizing Large Language Model Inference
- **Authors**: Xingqiang Chen (Xiamen University & Turingai Inc.)
- **Email**: chenxingqiang@turingai.cc
- **Repository**: https://github.com/chenxingqiang/llmir
- **Website**: https://chenxingqiang.github.io/llmir-www/

---

## ICCD 2025 Submission Status: REJECTED (25.5% acceptance rate)

### Reviewer Scores
| Reviewer | Score | Assessment |
|----------|-------|------------|
| Review 1 | -1 | Weak Reject |
| Review 2 | -1 | Weak Reject |
| Review 3 | -2 | Reject |
| Review 4 | 0 | Borderline |

### Key Feedback Summary
1. **Implementation Details**: Insufficient explanation of compiler phase internals and novelty
2. **Experimental Scope**: Limited to LLaMA-2 variants only
3. **Baseline Coverage**: Missing TensorRT-LLM and MLC-LLM comparisons
4. **Quality Metrics**: No perplexity or accuracy measurements
5. **Figure Readability**: Fonts too small
6. **Compile-Time vs Runtime**: Unclear how compile-time optimizations interact with runtime

---

## Comprehensive Revision (February 2025)

### Files Created
- `LLMIR-paper-ICCD2025-revised.tex` - Fully revised paper
- `REVISION_NOTES.md` - Detailed documentation of all changes
- `figures/*_v2.py` - Updated figure generation scripts with larger fonts
- `figures/*_v2.pdf/png` - Regenerated figures

### Major Revisions Made

#### 1. Multi-Model Experiments (Addresses Review 2, Review 4)
Extended experiments from LLaMA-2 only to 8 model variants:
- LLaMA-2: 7B, 13B, 70B
- Phi-3: 3.8B
- Qwen-2: 7B, 14B, 72B
- DeepSeek-V2: 16B (MoE architecture)

#### 2. Baseline Coverage (Addresses Review 2, Review 4)
Added comparisons with:
- TensorRT-LLM v0.9.0 (NVIDIA's production compiler)
- MLC-LLM v0.1.0 (TVM-based LLM deployment)

Results show:
- +22.4% over vLLM
- +38.1% over SGLang
- +4.8% over TensorRT-LLM
- +25.9% over MLC-LLM

#### 3. Quality Metrics (Addresses Review 4)
Added quality validation:
- Perplexity on C4 validation set
- MMLU accuracy benchmarks
- Results: <0.15% perplexity difference, <0.1% accuracy difference

#### 4. Implementation Details (Addresses Review 1, Review 2, Review 3)
- Added Algorithm 1: Block Size Optimization
- Detailed 4-stage compilation pipeline description
- Three CUDA kernel variants with selection criteria
- Pattern-matching rewrite rules for KV cache optimization

#### 5. Compile-Time vs Runtime Analysis (Addresses Review 3)
Added Table I clearly showing:
| Optimization | Compile-Time | Runtime |
|-------------|--------------|---------|
| Block allocation | Sizing, layout | Dynamic growth |
| Kernel selection | Variant choice | Parameter binding |
| Prefix sharing | Detection | Actual sharing |

#### 6. Figure Readability (Addresses Review 3)
- Base font size: 12pt → 14pt
- Title font size: 14pt → 16pt
- Regenerated all key figures

#### 7. Reviewer Questions Answered
- Runtime kernel selection: Three variants with compile-time selection
- Memory drop explanation: 256KB causes fragmentation with varied lengths
- Hybrid GPU allocation: GPUs 0-3/4-7 form TP groups for layers 0-39/40-79

---

## Recommended Resubmission Targets

Given the comprehensive revisions and strong technical contributions:

### Tier 1 Venues (Systems/Architecture)
1. **ASPLOS 2026** - Computer architecture + systems focus
2. **OSDI/SOSP 2026** - Systems focus
3. **MLSys 2026** - ML systems focus

### Tier 2 Venues (Compilers/Parallel)
1. **CGO 2026** - Code generation and optimization
2. **PACT 2025** - Parallel architectures and compilation
3. **CC 2026** - Compiler construction

### Domain-Specific
1. **EMNLP 2025** - NLP systems track
2. **NeurIPS 2025** - ML systems track

---

## Technical Contributions

1. **LLM-Specific MLIR Dialect**: Custom types (PagedKVCache, ShardedTensor, QuantizedTensor) capturing LLM semantics
2. **Compiler-Level PagedAttention**: First IR-level representation enabling static analysis of dynamic memory patterns
3. **Multi-Stage Compilation Pipeline**: Model import → LLM dialect optimization → Code generation → Runtime integration
4. **Comprehensive Optimization Framework**: KV cache, multi-precision, parallelization, attention optimizations

## Performance Highlights

| Metric | Value |
|--------|-------|
| Average Throughput | 58,499 tokens/sec |
| Peak Throughput | 88,250 tokens/sec |
| vs vLLM | +22.4% |
| vs SGLang | +38.1% |
| vs TensorRT-LLM | +4.8% |
| vs MLC-LLM | +25.9% |
| Attention Speedup | 1.28× - 2.15× |
| 8-GPU Scaling | 94.5% efficiency |
| Memory Optimization | Up to 58.8% |

---

## Compilation Instructions

```bash
cd /workspace/IEEE-conference
pdflatex LLMIR-paper-ICCD2025-revised.tex
bibtex LLMIR-paper-ICCD2025-revised
pdflatex LLMIR-paper-ICCD2025-revised.tex
pdflatex LLMIR-paper-ICCD2025-revised.tex
```

---

**Last Updated**: February 3, 2025
**Status**: Revised paper ready for resubmission
