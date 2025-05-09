add_mlir_dialect_library(MLIRLLMDialect
  LLMDialect.cpp
  LLMOps.cpp
  LLMTypes.cpp

  DEPENDS
  MLIRLLMOpsIncGen
  MLIRLLMTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRTensorDialect
) 