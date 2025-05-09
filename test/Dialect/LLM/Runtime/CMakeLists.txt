add_mlir_unittest(MLIRLLMRuntimeTests
  kv_cache_test.cpp
)

target_link_libraries(MLIRLLMRuntimeTests PRIVATE
  MLIRLLMRuntime
  MLIRIR
  MLIRSupport
) 