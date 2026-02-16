# Phase 7 Design Document

## Overview

Phase 7 focuses on production readiness and ecosystem integration: HuggingFace Transformers, integration tests with real models, distributed training support, and deployment automation.

## Design Principles

- **Minimal Change**: Extend existing modules; create new files only when architecturally justified
- **Optional Dependencies**: HuggingFace/Transformers remain optional via `llmir[full]`
- **No Mocking**: Integration tests run against real models (config-only when possible to avoid heavy downloads)

---

## 1. HuggingFace Transformers Integration

### Objective

Enable LLMIR to auto-configure KV cache and optimizations from any HuggingFace model identifier, without requiring users to manually map model configs.

### Design

**Interface**:
```python
# New module: llmir.integration.huggingface
from llmir.integration.huggingface import from_pretrained

optimizer = from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
kv_config = optimizer.get_optimized_kv_cache_config()
```

**Implementation**:
1. Load `AutoConfig.from_pretrained(model_id)` (config only, no weights)
2. Map HF config attributes to `ModelConfig`:
   - `num_hidden_layers` → `num_layers`
   - `num_attention_heads` → `num_attention_heads`
   - `num_key_value_heads` → `num_key_value_heads` (GQA)
   - `hidden_size`, `intermediate_size`, `vocab_size`, `max_position_embeddings`
   - `rope_theta`, `rope_scaling_factor` (Llama 3.x)
   - `sliding_window` (Mistral)
3. Detect `model_type` (e.g. `llama`, `mistral`, `phi`) and return appropriate optimizer subclass
4. Fallback to `ModelOptimizer(config)` for unknown types

**Supported Architectures** (initial):
- Llama / Llama2 / Llama3
- Mistral / Mixtral
- Phi-2 / Phi-3
- Qwen / Qwen2 (via generic path)
- Gemma (via generic path)

**File**: `src/llmir/integration/__init__.py`, `src/llmir/integration/huggingface.py`

**Dependency**: `transformers>=4.30.0` (optional, in `full` extra)

---

## 2. Integration Tests with Actual LLM Models

### Objective

Validate LLMIR optimizations against real model configs; optionally run inference when dependencies available.

### Design

**Scope**:
- Config-only tests: Load HF config → LLMIR optimizer → verify KV config consistency (no weights)
- Optional inference test: Requires `llmir[full]`, skips if transformers not installed

**Test Structure**:
```
tests/
  test_integration_hf.py    # HF config → ModelOptimizer (requires network)
  test_integration_config.py # Config round-trip, memory estimates (no network) ✅
```

**Config-Only Flow** (always runnable with `llmir[full]`):
1. `from_pretrained("meta-llama/Llama-3.1-8B")` → optimizer
2. `optimizer.get_optimized_kv_cache_config()` → KVCacheConfig
3. Assert `num_layers`, `num_heads`, `head_dim` match expected
4. `ModelMemoryEstimator(config).estimate_total_memory(1, 128)` → sanity check

**Markers**:
- `@pytest.mark.requires_transformers` for HF-dependent tests
- `@pytest.mark.slow` for tests that download configs from Hub

---

## 3. StructAttr Migration (MLIR 18 Compatibility)

### Objective

Migrate KVCache.td from deprecated StructAttr to Properties-based attributes for MLIR 18 compatibility.

### Design

**Current**: `LLM_KVCacheConfig`, `LLM_KVCacheStats` use `StructAttr`

**Target**: Use `AttrDef` with `storageType` or dialect attributes with explicit storage. Reference: [MLIR Defining Dialects - Attributes](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/).

**Tasks**:
1. ~~Remove KVCache.td include from LLMKVCacheOps.td~~ DONE - ops don't use those attrs
2. Audit `include/mlir/Dialect/LLM/Runtime/KVCache.td` for Runtime lib usage
3. Replace StructAttr with AttrDef in KVCache.td when Runtime needs them
4. Update `build_llm_dialect/CMakeLists.txt` to re-enable LLMKVCacheOps

**Priority**: Medium (blocks standalone LLM dialect build on MLIR 18+)

---

## 4. Distributed Training Support (Future)

### Scope (Design Only)

- Integrate with PyTorch Distributed / FSDP for KV cache state sync
- Support checkpoint resumption across ranks
- Document usage with `torchrun` / `accelerate`

**Status**: Deferred; requires detailed design in separate doc.

---

## 5. Auto-Scaling and Kubernetes (Future)

### Scope (Design Only)

- Health checks and readiness probes for serving
- Horizontal Pod Autoscaler (HPA) integration
- Resource requests/limits recommendations

**Status**: Deferred; requires operational requirements from stakeholders.

---

## Implementation Order

| Order | Task | Effort | Dependency |
|-------|------|--------|------------|
| 1 | HuggingFace Integration | 1-2 days | llmir.models |
| 2 | Integration Tests (config-only) | 0.5 day | #1 |
| 3 | StructAttr Migration | 2-3 days | MLIR 18 docs |
| 4 | Optional inference test | 0.5 day | #1, #2 |

**Recommended Start**: Task 1 (HuggingFace Integration).
