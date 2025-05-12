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

# Check for Metal support on Apple platforms
if(LLMIR_ENABLE_METAL AND APPLE)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  target_link_libraries(llama3_kvcache_benchmark PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
  target_compile_definitions(llama3_kvcache_benchmark PRIVATE LLMIR_ENABLE_METAL)
endif()

install(TARGETS llama3_kvcache_benchmark
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
) 