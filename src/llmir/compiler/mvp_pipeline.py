"""MVP-A: single-layer append_kv → lookup_kv → paged_attention compile path."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from llmir.compiler.block_size import analyze_block_size, optimize_block_size_attr
from llmir.compiler.kv_emit import KVMicroPipelineConfig, emit_kv_micro_pipeline_mlir
from llmir.compiler.opt_driver import OptResult, run_mlir_opt
from llmir.compiler.pipeline import compile_kv_micro_pipeline


@dataclass
class MVPSingleLayerResult:
    """Artifacts from the paper-aligned single-layer e2e path."""

    mlir_before: str
    mlir_after_block_size: str
    block_size_before: int
    block_size_after: int
    block_analysis: Dict[str, Any]
    opt_optimize: Optional[OptResult] = None
    opt_lower: Optional[OptResult] = None
    lowered_mlir: str = ""
    reference_backend: str = ""
    torch_max_abs_diff: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _patch_block_size(mlir_text: str, new_size: int) -> str:
    return re.sub(
        r"block_size\s*=\s*\d+\s*:\s*i32",
        f"block_size = {new_size} : i32",
        mlir_text,
        count=1,
    )


def _read_block_size(mlir_text: str) -> int:
    match = re.search(r"block_size\s*=\s*(\d+)\s*:\s*i32", mlir_text)
    if not match:
        raise ValueError("block_size attribute not found in MLIR")
    return int(match.group(1))


def run_mvp_single_layer_e2e(
    cfg: Optional[KVMicroPipelineConfig] = None,
    *,
    oversized_block_size: int = 1024,
    run_mlir_passes: bool = True,
    run_reference: bool = True,
    compare_torch: bool = False,
    seed: int = 0,
) -> MVPSingleLayerResult:
    """
    Paper MVP-A pipeline:

    1. Emit single-layer ``append_kv / lookup_kv / paged_attention`` MLIR
    2. Apply compile-time block size analysis (Algorithm 1)
    3. Optionally run ``-llm-optimize-kv-cache`` and ``-llm-lower-kv-cache-ops``
    4. Execute Python/native reference and optional torch check
    """
    base_cfg = cfg or KVMicroPipelineConfig()
    emit_cfg = KVMicroPipelineConfig(
        num_layers=base_cfg.num_layers,
        num_heads=base_cfg.num_heads,
        head_dim=base_cfg.head_dim,
        block_size=oversized_block_size,
        max_seq_len=base_cfg.max_seq_len,
        batch_size=base_cfg.batch_size,
        seq_len=base_cfg.seq_len,
        dtype=base_cfg.dtype,
    )
    mlir_before = emit_kv_micro_pipeline_mlir(emit_cfg)
    analysis = analyze_block_size([emit_cfg.seq_len])
    optimal = optimize_block_size_attr(oversized_block_size, [emit_cfg.seq_len])
    mlir_after = _patch_block_size(mlir_before, optimal)

    result = MVPSingleLayerResult(
        mlir_before=mlir_before,
        mlir_after_block_size=mlir_after,
        block_size_before=_read_block_size(mlir_before),
        block_size_after=optimal,
        block_analysis=asdict(analysis),
    )

    mlir_for_opt = mlir_after
    if run_mlir_passes:
        result.opt_optimize = run_mlir_opt(
            mlir_after, passes=("-llm-optimize-kv-cache",)
        )
        if result.opt_optimize.success:
            mlir_for_opt = result.opt_optimize.stdout
            result.metadata["mlir_opt_block_size"] = _read_block_size(mlir_for_opt)

        result.opt_lower = run_mlir_opt(
            mlir_for_opt,
            passes=("-llm-optimize-kv-cache", "-llm-lower-kv-cache-ops"),
        )
        if result.opt_lower.success:
            result.lowered_mlir = result.opt_lower.stdout
            result.metadata["lowered_has_runtime_calls"] = (
                "mlir_llm_append_kv" in result.lowered_mlir
            )

    if run_reference or compare_torch:
        tuned_cfg = KVMicroPipelineConfig(
            num_layers=base_cfg.num_layers,
            num_heads=base_cfg.num_heads,
            head_dim=base_cfg.head_dim,
            block_size=optimal,
            max_seq_len=base_cfg.max_seq_len,
            batch_size=base_cfg.batch_size,
            seq_len=base_cfg.seq_len,
            dtype=base_cfg.dtype,
        )
        ref = compile_kv_micro_pipeline(
            tuned_cfg,
            run_opt=False,
            run_reference=run_reference or compare_torch,
            compare_torch=compare_torch,
            seed=seed,
        )
        result.reference_backend = ref.reference_backend
        result.torch_max_abs_diff = ref.torch_max_abs_diff
        result.metadata.update(ref.metadata)

    result.metadata["pipeline"] = "mvp-a-single-layer"
    result.metadata["paper_algorithm"] = "block_size_optimization_v1"
    return result


def mvp_result_to_json(result: MVPSingleLayerResult) -> str:
    payload = {
        "block_size_before": result.block_size_before,
        "block_size_after": result.block_size_after,
        "block_analysis": result.block_analysis,
        "reference_backend": result.reference_backend,
        "torch_max_abs_diff": result.torch_max_abs_diff,
        "mlir_opt_success": bool(result.opt_optimize and result.opt_optimize.success),
        "mlir_lower_success": bool(result.opt_lower and result.opt_lower.success),
        "metadata": result.metadata,
    }
    return json.dumps(payload, indent=2)
