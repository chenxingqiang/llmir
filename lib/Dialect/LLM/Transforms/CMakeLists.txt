# Placeholder for future transformation implementations 

add_mlir_dialect_library(MLIRLLMTransforms
  LLMLowering.cpp
  LowerKVCacheOps.cpp
  KVCacheOptimization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLM/Transforms

  DEPENDS
  MLIRLLMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLLMDialect
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRArithDialect
) 