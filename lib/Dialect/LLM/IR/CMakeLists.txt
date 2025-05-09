add_mlir_dialect_library(MLIRLLMDialect
  LLMDialect.cpp
  LLMOps.cpp
  LLMTypes.cpp
  LLMKVCacheOps.cpp
  LLMKVCacheOpsInterface.cpp

  DEPENDS
  MLIRLLMOpsIncGen
  MLIRLLMTypesIncGen
  MLIRLLMRuntimeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLLMRuntime
) 