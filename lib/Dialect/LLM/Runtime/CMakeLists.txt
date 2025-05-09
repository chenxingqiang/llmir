add_mlir_dialect_library(MLIRLLMRuntime
  KVCache.cpp

  DEPENDS
  MLIRLLMOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLLM
) 