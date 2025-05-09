//===- RuntimeInterfaces.h - MLIR LLM Runtime interfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines runtime interfaces for the LLM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLM_RUNTIME_RUNTIMEINTERFACES_H_
#define MLIR_DIALECT_LLM_RUNTIME_RUNTIMEINTERFACES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class DialectRegistry;

namespace llm {

// Forward declarations
class LLMDialect;

//===----------------------------------------------------------------------===//
// KVCacheInterface
//===----------------------------------------------------------------------===//

/// Interface for operations that interact with the KV cache.
class KVCacheInterface {
public:
  /// The interface ID.
  using InterfaceID = TypeID;
  static InterfaceID getInterfaceID() { return InterfaceID::get(); }

  /// Base requirement for an interface, for CRTP pattern.
  template <typename ConcreteType>
  class Interface;

  /// For direct subtype of OpInterface, require ConcreteType to be
  /// a subtype of ConcreteOp.
  template <typename ConcreteType, typename ConcreteOp>
  class ExternalModel;

  /// Default model for op providing the interface methods.
  template <typename ConcreteOp>
  class Model;

  /// Returns true if the operation uses a KV cache.
  virtual bool usesKVCache() = 0;

  /// Get the number of KV tokens processed by this operation.
  /// Returns -1 if not statically known.
  virtual int64_t getNumKVTokens() = 0;

  /// Get the KV cache input value, if any.
  virtual Value getKVCacheInput() = 0;

  /// Get the KV cache output value, if any.
  virtual Value getKVCacheOutput() = 0;

protected:
  /// Destructor marked as protected to ensure proper lifetime.
  virtual ~KVCacheInterface() = default;
};

//===----------------------------------------------------------------------===//
// AttentionInterface
//===----------------------------------------------------------------------===//

/// Interface for operations that perform attention computation.
class AttentionInterface {
public:
  /// The interface ID.
  using InterfaceID = TypeID;
  static InterfaceID getInterfaceID() { return InterfaceID::get(); }

  /// Base requirement for an interface, for CRTP pattern.
  template <typename ConcreteType>
  class Interface;

  /// For direct subtype of OpInterface, require ConcreteType to be
  /// a subtype of ConcreteOp.
  template <typename ConcreteType, typename ConcreteOp>
  class ExternalModel;

  /// Default model for op providing the interface methods.
  template <typename ConcreteOp>
  class Model;

  /// Returns true if the operation performs attention computation.
  virtual bool isAttentionOp() = 0;

  /// Get the batch size for this attention operation.
  /// Returns -1 if not statically known.
  virtual int64_t getBatchSize() = 0;

  /// Get the sequence length for this attention operation.
  /// Returns -1 if not statically known.
  virtual int64_t getSeqLength() = 0;

  /// Get the number of attention heads.
  virtual int64_t getNumHeads() = 0;

  /// Get the dimension of each attention head.
  virtual int64_t getHeadDim() = 0;

protected:
  /// Destructor marked as protected to ensure proper lifetime.
  virtual ~AttentionInterface() = default;
};

//===----------------------------------------------------------------------===//
// External Model Registration
//===----------------------------------------------------------------------===//

/// Register external models for the KVCacheInterface.
void registerKVCacheInterfaceExternalModels(DialectRegistry &registry);

/// Register external models for the AttentionInterface.
void registerAttentionInterfaceExternalModels(DialectRegistry &registry);

} // namespace llm
} // namespace mlir

// Include the generated interface declarations
#include "mlir/Dialect/LLM/Runtime/RuntimeInterfaces.h.inc"

#endif // MLIR_DIALECT_LLM_RUNTIME_RUNTIMEINTERFACES_H_ 