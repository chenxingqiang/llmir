add_mlir_dialect_library(MLIRLLMRuntime
  KVCache.cpp
  RuntimeInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLM/Runtime

  DEPENDS
  MLIRLLMRuntimeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLLMDialect
) 