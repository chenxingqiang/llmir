add_mlir_dialect_library(MLIRLLMRuntime
  KVCache.cpp
  KVCacheRuntime.cpp
  RuntimeInterfaces.cpp
  GPUMemoryUtils.cpp
  AttentionOpt.cpp
  MultiQueryAttention.cpp
  PrunedAttention.cpp
  QuantizedKVCache.cpp
  KVCacheSerialization.cpp
  DistributedKVCache.cpp
  SpeculativeKVCache.cpp
  PrefixCache.cpp
  AdaptiveBlockManager.cpp

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

# Check for Metal support on Apple platforms
if(LLMIR_ENABLE_METAL AND APPLE)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  target_link_libraries(MLIRLLMRuntime PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_METAL)
  message(STATUS "Building MLIRLLMRuntime with Metal support")
endif()

# Check for NCCL support (for distributed KV cache)
if(LLMIR_ENABLE_NCCL)
  find_package(NCCL REQUIRED)
  target_include_directories(MLIRLLMRuntime PRIVATE ${NCCL_INCLUDE_DIRS})
  target_link_libraries(MLIRLLMRuntime PRIVATE ${NCCL_LIBRARIES})
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_NCCL)
  message(STATUS "Building MLIRLLMRuntime with NCCL support for distributed cache")
endif() 