"""LLMIR compiler helpers (MLIR emit, opt driver, reference interpreter)."""

from llmir.compiler.interpreter import numpy_paged_attention, run_kv_micro_pipeline_reference
from llmir.compiler.kv_emit import KVMicroPipelineConfig, emit_kv_micro_pipeline_mlir
from llmir.compiler.opt_driver import OptResult, find_mlir_opt, run_mlir_opt
from llmir.compiler.block_size import BlockSizeAnalysisResult, analyze_block_size
from llmir.compiler.mvp_pipeline import MVPSingleLayerResult, run_mvp_single_layer_e2e
from llmir.compiler.pipeline import CompilePipelineResult, compile_kv_micro_pipeline

__all__ = [
    "BlockSizeAnalysisResult",
    "analyze_block_size",
    "MVPSingleLayerResult",
    "run_mvp_single_layer_e2e",
    "KVMicroPipelineConfig",
    "emit_kv_micro_pipeline_mlir",
    "find_mlir_opt",
    "run_mlir_opt",
    "OptResult",
    "numpy_paged_attention",
    "run_kv_micro_pipeline_reference",
    "compile_kv_micro_pipeline",
    "CompilePipelineResult",
]
