add_mlir_unittest(MLIRLLMRuntimeTests
  kv_cache_test.cpp
)

add_llvm_executable(kv_cache_test
  kv_cache_test.cpp
  kv_cache_unit_test.cpp
)

target_link_libraries(MLIRLLMRuntimeTests PRIVATE
  MLIRLLMRuntime
  MLIRIR
  MLIRSupport
)

target_link_libraries(kv_cache_test PRIVATE
  MLIRLLMRuntime
  gtest_main
) 