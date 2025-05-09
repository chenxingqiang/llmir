add_mlir_dialect_library(MLIRLLMRuntimeIncGen
  DEPENDS
  MLIRLLMOpsIncGen

  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)

set(LLVM_TARGET_DEFINITIONS KVCache.td)
mlir_tablegen(KVCache.h.inc -gen-struct-attr-decls)
mlir_tablegen(KVCache.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRLLMRuntimeIncGen)

add_mlir_interface(RuntimeInterfaces) 