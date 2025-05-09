# Placeholder for future transformation implementations 

add_mlir_dialect_library(MLIRLLMTransforms
  KVCacheOptimization.cpp

  DEPENDS
  MLIRLLMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRLLMDialect
) 