option(LLMIR_ENABLE_CUDA "Build LLMIR runtime CUDA kernels (CUDAKernels.cu)" OFF)
option(LLMIR_ENABLE_HIP "Build LLMIR runtime with HIP support" OFF)
option(LLMIR_ENABLE_METAL "Build LLMIR runtime with Metal support" OFF)
option(LLMIR_ENABLE_NCCL "Build LLMIR runtime with NCCL support" OFF)

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
  ContinuousBatching.cpp
  VLLMIntegration.cpp
  ModelOptimizations.cpp
  CApi.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLM/Runtime

  DEPENDS
  MLIRLLMRuntimeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLLMDialect
)

if(LLMIR_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(MLIRLLMRuntime PRIVATE CUDAKernels.cu)
  target_include_directories(MLIRLLMRuntime PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(MLIRLLMRuntime PRIVATE CUDA::cudart)
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_CUDA)
  set_target_properties(MLIRLLMRuntime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  message(STATUS "Building MLIRLLMRuntime with CUDA kernels (CUDAKernels.cu)")
endif()

if(LLMIR_ENABLE_HIP)
  find_package(hip REQUIRED)
  target_include_directories(MLIRLLMRuntime PRIVATE ${HIP_INCLUDE_DIRS})
  target_link_libraries(MLIRLLMRuntime PRIVATE hip::device)
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_HIP)
  message(STATUS "Building MLIRLLMRuntime with HIP support")
endif()

if(LLMIR_ENABLE_METAL AND APPLE)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  target_link_libraries(MLIRLLMRuntime PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_METAL)
  message(STATUS "Building MLIRLLMRuntime with Metal support")
endif()

if(LLMIR_ENABLE_NCCL)
  find_package(NCCL REQUIRED)
  target_include_directories(MLIRLLMRuntime PRIVATE ${NCCL_INCLUDE_DIRS})
  target_link_libraries(MLIRLLMRuntime PRIVATE ${NCCL_LIBRARIES})
  target_compile_definitions(MLIRLLMRuntime PRIVATE LLMIR_ENABLE_NCCL)
  message(STATUS "Building MLIRLLMRuntime with NCCL support for distributed cache")
endif()
