add_executable(llama3_kvcache_benchmark
  llama3_kvcache_benchmark.cpp
)

target_link_libraries(llama3_kvcache_benchmark PRIVATE
  MLIRLLMRuntime
  MLIRLLMDialect
  benchmark::benchmark
  benchmark::benchmark_main
)

# Check for CUDA support
if(LLMIR_ENABLE_CUDA)
  target_compile_definitions(llama3_kvcache_benchmark PRIVATE LLMIR_ENABLE_CUDA)
endif()

# Check for HIP support
if(LLMIR_ENABLE_HIP)
  target_compile_definitions(llama3_kvcache_benchmark PRIVATE LLMIR_ENABLE_HIP)
endif()

install(TARGETS llama3_kvcache_benchmark
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
) 