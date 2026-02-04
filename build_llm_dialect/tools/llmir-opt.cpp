//===- llmir-opt.cpp - LLMIR optimizer driver ----------------------------===//
//
// Main entry point for the LLMIR optimizer tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// Include LLMIR dialect headers
#include "mlir/Dialect/LLM/IR/LLM.h"
#include "mlir/Dialect/LLM/Transforms/Passes.h"

int main(int argc, char **argv) {
  // Register all MLIR dialects
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  
  // Register LLMIR dialect
  registry.insert<mlir::llm::LLMDialect>();
  
  // Register all passes
  mlir::registerAllPasses();
  
  // Register LLMIR passes
  mlir::llm::registerLLMPasses();
  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LLMIR optimizer driver\n", registry));
}
