"""E4: compositional / trace-driven verification (E1 + E2 + E3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmir.compiler.block_size import analyze_block_size, optimize_block_size_attr
from llmir.models import ModelRegistry


@dataclass(frozen=True)
class E4WorkloadTrace:
    """Shared-prefix decoder trace (S1/S2/S3 buckets)."""

    shared_prefix_tokens: int  # L_s
    num_requests: int  # N
    suffix_tokens: int  # L_u
    decode_steps: int = 4
    block_size_before: int = 1024
    block_size_after: int = 32
    model_preset: str = "qwen3-8b"


@dataclass
class E4CompositionalResult:
    """JSON-serializable E4 analysis."""

    experiment: str = "E4"
    mode: str = "compositional_verification"
    trace: Dict[str, Any] = field(default_factory=dict)
    e1_block_sizing: Dict[str, Any] = field(default_factory=dict)
    e2_prefix_prefill: Dict[str, Any] = field(default_factory=dict)
    e3_kv_host_copies: Dict[str, Any] = field(default_factory=dict)
    composite: Dict[str, Any] = field(default_factory=dict)
    measured_comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if out.get("measured_comparison") is None:
            out.pop("measured_comparison", None)
        return out


def _blocks_for_length(seq_len: int, block_size: int) -> int:
    if seq_len <= 0 or block_size <= 0:
        return 0
    return (seq_len + block_size - 1) // block_size


def _allocated_tokens(seq_len: int, block_size: int) -> int:
    return _blocks_for_length(seq_len, block_size) * block_size


def analyze_e1_block_sizing(trace: E4WorkloadTrace) -> Dict[str, Any]:
    """E1: compile-time block_size attr vs naive oversized attribute (Algorithm 1)."""
    seq_total = trace.shared_prefix_tokens + trace.suffix_tokens
    seq_lengths = [trace.shared_prefix_tokens, trace.suffix_tokens, seq_total]
    analysis = analyze_block_size(seq_lengths)
    e1_pass_block_size = optimize_block_size_attr(trace.block_size_before, seq_lengths)

    blocks_before = _blocks_for_length(seq_total, trace.block_size_before)
    blocks_after_pass = _blocks_for_length(seq_total, e1_pass_block_size)

    alloc_before = sum(
        _allocated_tokens(length, trace.block_size_before) for length in seq_lengths
    )
    alloc_after_pass = sum(
        _allocated_tokens(length, e1_pass_block_size) for length in seq_lengths
    )

    total_blocks_before = blocks_before * trace.num_requests
    total_blocks_after_pass = blocks_after_pass * trace.num_requests

    return {
        "seq_lengths": seq_lengths,
        "algorithm1_optimal_block_size": analysis.optimal_block_size,
        "block_size_attr_before": trace.block_size_before,
        "block_size_attr_after_e1_pass": e1_pass_block_size,
        "block_size_reduction_ratio": (
            1.0 - (e1_pass_block_size / trace.block_size_before)
            if trace.block_size_before
            else 0.0
        ),
        "blocks_per_request_before_attr": blocks_before,
        "blocks_per_request_after_e1_pass": blocks_after_pass,
        "total_kv_blocks_all_requests_before": total_blocks_before,
        "total_kv_blocks_all_requests_after_e1_pass": total_blocks_after_pass,
        "allocated_tokens_all_lengths_before": alloc_before,
        "allocated_tokens_all_lengths_after_e1_pass": alloc_after_pass,
        "allocated_tokens_reduction_ratio": (
            1.0 - (alloc_after_pass / alloc_before) if alloc_before else 0.0
        ),
        "fragmentation_score_at_optimal": analysis.fragmentation_score,
    }


def analyze_e2_prefix_prefill(trace: E4WorkloadTrace) -> Dict[str, Any]:
    """E2: shared-prefix prefill token accounting (analytical proxy)."""
    ls, n, lu = trace.shared_prefix_tokens, trace.num_requests, trace.suffix_tokens
    per_request_len = ls + lu

    # Cold: every request recomputes full prompt prefill.
    cold_prefill_tokens = n * per_request_len

    # Warm prefix: pay L_s once, then each request only suffix (+ optional decode).
    warm_prefill_tokens = ls + n * lu
    saved_prefill_tokens = cold_prefill_tokens - warm_prefill_tokens

    # KV-layer sim ideal: first request builds prefix KV, others append suffix only.
    ideal_kv_ops_baseline = n * per_request_len
    ideal_kv_ops_cached = per_request_len + (n - 1) * lu if n > 0 else 0
    ideal_kv_speedup = (
        ideal_kv_ops_baseline / ideal_kv_ops_cached if ideal_kv_ops_cached > 0 else 1.0
    )

    return {
        "cold_prefill_tokens_total": cold_prefill_tokens,
        "warm_prefill_tokens_total": warm_prefill_tokens,
        "saved_prefill_tokens": saved_prefill_tokens,
        "prefill_reduction_ratio": (
            saved_prefill_tokens / cold_prefill_tokens if cold_prefill_tokens else 0.0
        ),
        "ideal_kv_layer_speedup_upper_bound": ideal_kv_speedup,
    }


def analyze_e3_host_copies(trace: E4WorkloadTrace) -> Dict[str, Any]:
    """E3: host↔device KV round-trips per decode step (integration proxy)."""
    steps_per_request = max(0, trace.decode_steps)
    total_decode_steps = steps_per_request * trace.num_requests

    # numpy PagedKV path: host staging each step; torch_cuda: 0 host copies on hot path.
    numpy_round_trips_per_step = 2
    torch_cuda_round_trips_per_step = 0

    return {
        "decode_steps_per_request": steps_per_request,
        "total_decode_steps": total_decode_steps,
        "numpy_host_round_trips_total": numpy_round_trips_per_step * total_decode_steps,
        "torch_cuda_host_round_trips_total": torch_cuda_round_trips_per_step
        * total_decode_steps,
        "host_copy_reduction_ratio": 1.0 if total_decode_steps else 0.0,
    }


def _kv_memory_bytes(
    *,
    num_layers: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    bytes_per_element: int = 2,
) -> int:
    # K + V per layer per block slot
    per_block = 2 * kv_heads * block_size * head_dim * bytes_per_element
    return num_layers * num_blocks * per_block


def build_composite_summary(
    trace: E4WorkloadTrace,
    e1: Dict[str, Any],
    e2: Dict[str, Any],
    e3: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine E1–E3 into a single compile-time value narrative."""
    registry = ModelRegistry()
    cfg = registry.get(trace.model_preset)
    arch_note = trace.model_preset
    kv_meta: Dict[str, Any] = {"model_preset": arch_note}
    if cfg:
        kv_meta.update(
            {
                "architecture": cfg.architecture.name,
                "num_layers": cfg.num_layers,
                "num_kv_heads": cfg.num_key_value_heads or cfg.num_attention_heads,
                "head_dim": cfg.get_head_dim(),
            }
        )
        blocks_after_pass = e1["blocks_per_request_after_e1_pass"]
        e1_block_size = e1["block_size_attr_after_e1_pass"]
        mem_before = _kv_memory_bytes(
            num_layers=cfg.num_layers,
            kv_heads=cfg.num_key_value_heads or cfg.num_attention_heads,
            head_dim=cfg.get_head_dim(),
            block_size=trace.block_size_before,
            num_blocks=e1["blocks_per_request_before_attr"] * trace.num_requests,
        )
        mem_after = _kv_memory_bytes(
            num_layers=cfg.num_layers,
            kv_heads=cfg.num_key_value_heads or cfg.num_attention_heads,
            head_dim=cfg.get_head_dim(),
            block_size=e1_block_size,
            num_blocks=blocks_after_pass * trace.num_requests,
        )
        kv_meta["estimated_kv_bytes_before_blocks"] = mem_before
        kv_meta["estimated_kv_bytes_after_blocks"] = mem_after
        kv_meta["kv_bytes_reduction_ratio"] = (
            1.0 - (mem_after / mem_before) if mem_before else 0.0
        )

    return {
        "claim_scope": (
            "Analytical compositional bounds for compile-time block/prefix/KV "
            "integration; not end-to-end vs vLLM throughput."
        ),
        "model": kv_meta,
        "compile_time_levers": {
            "e1_block_size_reduction": e1["block_size_reduction_ratio"],
            "e1_allocated_tokens_reduction": e1["allocated_tokens_reduction_ratio"],
            "e2_prefill_token_reduction": e2["prefill_reduction_ratio"],
            "e3_host_copy_elimination": e3["host_copy_reduction_ratio"],
        },
    }


def compare_to_measured_sim(
    e2_analytical: Dict[str, Any],
    sim_path: Path,
) -> Dict[str, Any]:
    """Optional: compare E2 ideal speedup to E2 KV simulation JSON."""
    with sim_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    results: List[Dict[str, Any]] = payload.get("results", [])
    baseline = next(
        (r for r in results if "baseline" in r.get("scenario", "").lower()),
        None,
    )
    cached = next(
        (
            r
            for r in results
            if "prefix_cached" in r.get("scenario", "").lower()
            or "cached" in r.get("scenario", "").lower()
        ),
        None,
    )
    measured_speedup = None
    if baseline and cached and cached.get("avg_latency_ms", 0) > 0:
        measured_speedup = baseline["avg_latency_ms"] / cached["avg_latency_ms"]
    ideal = e2_analytical.get("ideal_kv_layer_speedup_upper_bound", 1.0)
    return {
        "sim_json": str(sim_path),
        "ideal_kv_speedup_model": ideal,
        "measured_kv_sim_speedup": measured_speedup,
        "measured_within_ideal_bound": (
            measured_speedup <= ideal + 1e-6 if measured_speedup is not None else None
        ),
        "config": payload.get("config"),
    }


def run_e4_compositional_verification(
    trace: E4WorkloadTrace,
    *,
    measured_sim_json: Optional[Path] = None,
) -> E4CompositionalResult:
    """Full E4 pipeline."""
    e1 = analyze_e1_block_sizing(trace)
    e2 = analyze_e2_prefix_prefill(trace)
    e3 = analyze_e3_host_copies(trace)
    composite = build_composite_summary(trace, e1, e2, e3)

    measured = None
    if measured_sim_json is not None and measured_sim_json.is_file():
        measured = compare_to_measured_sim(e2, measured_sim_json)

    return E4CompositionalResult(
        trace=asdict(trace),
        e1_block_sizing=e1,
        e2_prefix_prefill=e2,
        e3_kv_host_copies=e3,
        composite=composite,
        measured_comparison=measured,
    )


def trace_from_sim_json(path: Path) -> E4WorkloadTrace:
    """Build trace from shared_prefix_decoder_*_sim.json."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("config", {})
    return E4WorkloadTrace(
        shared_prefix_tokens=int(cfg.get("system_prompt_tokens", 128)),
        num_requests=int(cfg.get("num_requests", 32)),
        suffix_tokens=int(cfg.get("user_suffix_tokens", 8)),
        decode_steps=int(cfg.get("max_new_tokens", 4)),
        model_preset=str(cfg.get("model", "qwen3-8b")),
    )


def run_e4_multi_bucket_verification(benchmarks_dir: Path) -> Dict[str, Any]:
    """Run E4 compositional analysis for all S1/S2/S3 decoder workload buckets."""
    from llmir.benchmark.decoder_workload_buckets import list_decoder_workload_buckets

    buckets: List[Dict[str, Any]] = []
    for bucket in list_decoder_workload_buckets():
        sim_path = bucket.artifact_path(benchmarks_dir)
        trace = trace_from_sim_json(sim_path)
        result = run_e4_compositional_verification(trace, measured_sim_json=sim_path)
        buckets.append(
            {
                "bucket_id": bucket.bucket_id,
                "bucket_label": bucket.label,
                "sim_json": str(sim_path),
                "analysis": result.to_dict(),
            }
        )
    return {
        "experiment": "E4",
        "mode": "multi_bucket_compositional",
        "benchmarks_dir": str(benchmarks_dir),
        "buckets": buckets,
    }
