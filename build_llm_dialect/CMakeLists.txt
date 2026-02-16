cmake_minimum_required(VERSION 3.20)
project(llmir-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM
find_package(LLVM REQUIRED CONFIG
    PATHS /usr/lib/llvm-18/cmake /usr/lib/llvm-18/lib/cmake/llvm)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find MLIR
find_package(MLIR REQUIRED CONFIG
    PATHS /usr/lib/llvm-18/lib/cmake/mlir)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/../include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Definitions
add_definitions(${LLVM_DEFINITIONS})

# Set up tablegen
set(LLVM_TABLEGEN_EXE /usr/bin/llvm-tblgen-18)
set(MLIR_TABLEGEN_EXE /usr/bin/mlir-tblgen-18)

# Source directories
set(LLMIR_SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)
set(LLMIR_BINARY_DIR ${PROJECT_BINARY_DIR})

# Create include directory
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/mlir/Dialect/LLM/IR)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/mlir/Dialect/LLM/Transforms)

# LLM Dialect TableGen
set(LLVM_TARGET_DEFINITIONS ${LLMIR_SOURCE_DIR}/include/mlir/Dialect/LLM/IR/LLMOps.td)
mlir_tablegen(include/mlir/Dialect/LLM/IR/LLMOps.h.inc -gen-op-decls)
mlir_tablegen(include/mlir/Dialect/LLM/IR/LLMOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/mlir/Dialect/LLM/IR/LLMOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/mlir/Dialect/LLM/IR/LLMOpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRLLMOpsIncGen)

# LLM Types TableGen
set(LLVM_TARGET_DEFINITIONS ${LLMIR_SOURCE_DIR}/include/mlir/Dialect/LLM/IR/LLMTypes.td)
mlir_tablegen(include/mlir/Dialect/LLM/IR/LLMTypes.h.inc -gen-typedef-decls)
mlir_tablegen(include/mlir/Dialect/LLM/IR/LLMTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRLLMTypesIncGen)

# Note: LLM KV Cache Ops uses StructAttr (via KVCache.td) removed in MLIR 18.
# KVCache.td include removed from LLMKVCacheOps.td - re-enable when building
# with full MLIR that has LLMKVCacheOpsIncGen. Standalone build skips KV cache ops.

# Passes TableGen
set(LLVM_TARGET_DEFINITIONS ${LLMIR_SOURCE_DIR}/include/mlir/Dialect/LLM/Transforms/Passes.td)
mlir_tablegen(include/mlir/Dialect/LLM/Transforms/Passes.h.inc -gen-pass-decls -name LLM)
add_public_tablegen_target(MLIRLLMPassesIncGen)

# LLM Dialect IR Library - use standard add_library instead of MLIR macros
add_library(MLIRLLMIR STATIC
    ${LLMIR_SOURCE_DIR}/lib/Dialect/LLM/IR/LLMDialect.cpp
    ${LLMIR_SOURCE_DIR}/lib/Dialect/LLM/IR/LLMOps.cpp
    ${LLMIR_SOURCE_DIR}/lib/Dialect/LLM/IR/LLMTypes.cpp
    # LLMKVCacheOps excluded - requires KVCache.td/StructAttr migration
)

add_dependencies(MLIRLLMIR
    MLIRLLMOpsIncGen
    MLIRLLMTypesIncGen
)

target_include_directories(MLIRLLMIR PUBLIC
    ${LLMIR_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

target_link_libraries(MLIRLLMIR PUBLIC
    MLIRIR
    MLIRSupport
)

# LLM Transforms Library
add_library(MLIRLLMTransforms STATIC
    ${LLMIR_SOURCE_DIR}/lib/Dialect/LLM/Transforms/KVCacheOptimization.cpp
)

add_dependencies(MLIRLLMTransforms
    MLIRLLMPassesIncGen
    MLIRLLMIR
)

target_include_directories(MLIRLLMTransforms PUBLIC
    ${LLMIR_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

target_link_libraries(MLIRLLMTransforms PUBLIC
    MLIRLLMIR
    MLIRIR
    MLIRPass
    MLIRTransforms
)

# Note: llmir-opt tool requires full MLIR library linkage
# Skipping for this standalone build verification

message(STATUS "LLMIR Dialect build configured successfully")
