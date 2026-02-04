cmake_minimum_required(VERSION 3.20)
project(llmir-llm-dialect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM and MLIR
find_package(LLVM 18 REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Include LLVM/MLIR directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Add LLVM definitions
add_definitions(${LLVM_DEFINITIONS})

# Include our headers
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_BINARY_DIR}/include)

# LLMIR LLM Dialect Library
add_library(LLMDialect STATIC
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/IR/LLMDialect.cpp
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/IR/LLMOps.cpp
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/IR/LLMTypes.cpp
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/IR/LLMKVCacheOps.cpp
)

target_link_libraries(LLMDialect
    MLIRSupport
    MLIRIR
    MLIRDialect
)

# LLMIR Runtime Library
add_library(LLMRuntime STATIC
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/Runtime/AttentionOpt.cpp
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/Runtime/KVCache.cpp
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/Runtime/PrefixCache.cpp
    ${CMAKE_SOURCE_DIR}/../lib/Dialect/LLM/Runtime/QuantizedKVCache.cpp
)

target_link_libraries(LLMRuntime
    MLIRSupport
)

# Simple test executable
add_executable(llm-dialect-test
    test_llm_dialect.cpp
)

target_link_libraries(llm-dialect-test
    LLMDialect
    LLMRuntime
    MLIRSupport
    MLIRIR
    MLIRDialect
    LLVMSupport
)

message(STATUS "LLMIR LLM Dialect build configured")
