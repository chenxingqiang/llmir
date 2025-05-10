add_mlir_dialect_library(MLIRLLMRuntime
  KVCache.cpp
  KVCacheRuntime.cpp
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

# Check for CUDA support
if(LLMIR_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  target_include_directories(MLIRLLMRuntime PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(MLIRLLMRuntime PRIVATE ${CUDA_LIBRARIES})
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_CUDA)
  message(STATUS "Building MLIRLLMRuntime with CUDA support")
endif()

# Check for HIP support
if(LLMIR_ENABLE_HIP)
  find_package(hip REQUIRED)
  target_include_directories(MLIRLLMRuntime PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(MLIRLLMRuntime PRIVATE hip::device)
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_HIP)
  message(STATUS "Building MLIRLLMRuntime with HIP support")
endif() 