add_mlir_dialect(LLMOps llm)
add_mlir_dialect(LLMTypes llm)

add_mlir_doc(LLM LLMDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(LLMOps LLMOps Dialects/ -gen-op-doc)
add_mlir_doc(LLMTypes LLMTypes Dialects/ -gen-typedef-doc) 