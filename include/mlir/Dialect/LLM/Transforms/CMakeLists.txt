# Placeholder for future transformations 

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLMTransforms)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix LLMTransforms)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix LLMTransforms)
add_public_tablegen_target(MLIRLLMTransformsIncGen)

add_mlir_doc(Passes LLMTransforms Dialects/ -gen-pass-doc) 