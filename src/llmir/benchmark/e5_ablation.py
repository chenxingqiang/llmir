"""E5: ablation at verifiable layers (E1 / E2 / E3 switches)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from llmir.benchmark.e4_compositional import (
    E4WorkloadTrace,
    analyze_e1_block_sizing,
    analyze_e2_prefix_prefill,
    analyze_e3_host_copies,
    trace_from_sim_json,
)


@dataclass(frozen=True)
class E5AblationSwitches:
    """Toggle compile-time / serving proxies independently."""

    block_opt: bool = True
    prefix_cache: bool = True
    torch_cuda_kv: bool = True

    def to_dict(self) -> Dict[str, bool]:
        return {
            "block_opt": self.block_opt,
            "prefix_cache": self.prefix_cache,
            "torch_cuda_kv": self.torch_cuda_kv,
        }


@dataclass(frozen=True)
class E5AblationPreset:
    """Named switch configuration for reproducible ablation rows."""

    name: str
    switches: E5AblationSwitches
    kind: str  # "baseline" | "isolated" | "cumulative" | "full"


# Standard matrix: isolated knobs + cumulative stack (paper Table ablation style).
DEFAULT_PRESETS: tuple[E5AblationPreset, ...] = (
    E5AblationPreset("baseline", E5AblationSwitches(False, False, False), "baseline"),
    E5AblationPreset(
        "e1_block_opt_only",
        E5AblationSwitches(True, False, False),
        "isolated",
    ),
    E5AblationPreset(
        "e2_prefix_only",
        E5AblationSwitches(False, True, False),
        "isolated",
    ),
    E5AblationPreset(
        "e3_gpu_kv_only",
        E5AblationSwitches(False, False, True),
        "isolated",
    ),
    E5AblationPreset(
        "cumulative_e1",
        E5AblationSwitches(True, False, False),
        "cumulative",
    ),
    E5AblationPreset(
        "cumulative_e1_e2",
        E5AblationSwitches(True, True, False),
        "cumulative",
    ),
    E5AblationPreset("full", E5AblationSwitches(True, True, True), "full"),
)


@dataclass
class E5AblationRow:
    """One configuration row with proxies and deltas."""

    name: str
    kind: str
    switches: Dict[str, bool]
    proxies: Dict[str, Any]
    delta_vs_baseline: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class E5AblationResult:
    """JSON-serializable E5 ablation output."""

    experiment: str = "E5"
    mode: str = "ablation_at_verifiable_layers"
    trace: Dict[str, Any] = field(default_factory=dict)
    configurations: List[Dict[str, Any]] = field(default_factory=list)
    isolated_contributions: Dict[str, Any] = field(default_factory=dict)
    cumulative_stack: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _e1_disabled(trace: E4WorkloadTrace) -> Dict[str, Any]:
    """E1 pass off: keep naive oversized block_size attr."""
    full = analyze_e1_block_sizing(trace)
    before = full["block_size_attr_before"]
    return {
        **full,
        "block_size_attr_after_e1_pass": before,
        "block_size_reduction_ratio": 0.0,
        "blocks_per_request_after_e1_pass": full["blocks_per_request_before_attr"],
        "total_kv_blocks_all_requests_after_e1_pass": full[
            "total_kv_blocks_all_requests_before"
        ],
        "allocated_tokens_all_lengths_after_e1_pass": full[
            "allocated_tokens_all_lengths_before"
        ],
        "allocated_tokens_reduction_ratio": 0.0,
        "pass_enabled": False,
    }


def _e2_disabled(trace: E4WorkloadTrace) -> Dict[str, Any]:
    """E2 prefix off: every request pays full cold prefill."""
    ls, n, lu = trace.shared_prefix_tokens, trace.num_requests, trace.suffix_tokens
    per_request_len = ls + lu
    cold_prefill_tokens = n * per_request_len
    return {
        "cold_prefill_tokens_total": cold_prefill_tokens,
        "warm_prefill_tokens_total": cold_prefill_tokens,
        "saved_prefill_tokens": 0,
        "prefill_reduction_ratio": 0.0,
        "ideal_kv_layer_speedup_upper_bound": 1.0,
        "prefix_cache_enabled": False,
    }


def _e3_disabled(trace: E4WorkloadTrace) -> Dict[str, Any]:
    """E3 off: NumPy host-staging path (no GPU-resident KV)."""
    base = analyze_e3_host_copies(trace)
    return {
        **base,
        "active_host_round_trips_total": base["numpy_host_round_trips_total"],
        "host_copy_reduction_ratio": 0.0,
        "torch_cuda_kv_enabled": False,
    }


def _e3_enabled(trace: E4WorkloadTrace) -> Dict[str, Any]:
    base = analyze_e3_host_copies(trace)
    return {
        **base,
        "active_host_round_trips_total": base["torch_cuda_host_round_trips_total"],
        "host_copy_reduction_ratio": 1.0,
        "torch_cuda_kv_enabled": True,
    }


def analyze_layer_proxies(
    trace: E4WorkloadTrace,
    switches: E5AblationSwitches,
) -> Dict[str, Any]:
    """Compute E1–E3 proxy metrics under the given switches."""
    e1 = analyze_e1_block_sizing(trace) if switches.block_opt else _e1_disabled(trace)
    if switches.block_opt:
        e1 = {**e1, "pass_enabled": True}

    e2 = (
        analyze_e2_prefix_prefill(trace)
        if switches.prefix_cache
        else _e2_disabled(trace)
    )
    if switches.prefix_cache:
        e2 = {**e2, "prefix_cache_enabled": True}

    e3 = _e3_enabled(trace) if switches.torch_cuda_kv else _e3_disabled(trace)

    return {
        "block_size_reduction_ratio": e1["block_size_reduction_ratio"],
        "allocated_tokens_reduction_ratio": e1["allocated_tokens_reduction_ratio"],
        "prefill_reduction_ratio": e2["prefill_reduction_ratio"],
        "saved_prefill_tokens": e2["saved_prefill_tokens"],
        "ideal_kv_layer_speedup_upper_bound": e2["ideal_kv_layer_speedup_upper_bound"],
        "host_copy_reduction_ratio": e3["host_copy_reduction_ratio"],
        "host_round_trips_total": e3["active_host_round_trips_total"],
        "e1_block_sizing": e1,
        "e2_prefix_prefill": e2,
        "e3_kv_host_copies": e3,
    }


_PROXY_KEYS = (
    "block_size_reduction_ratio",
    "allocated_tokens_reduction_ratio",
    "prefill_reduction_ratio",
    "saved_prefill_tokens",
    "ideal_kv_layer_speedup_upper_bound",
    "host_copy_reduction_ratio",
    "host_round_trips_total",
)


def _proxy_slice(proxies: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: proxies[key] for key in _PROXY_KEYS}


def _delta_vs_baseline(
    proxies: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for key in _PROXY_KEYS:
        if key == "host_round_trips_total":
            delta[key] = baseline[key] - proxies[key]
        else:
            delta[key] = proxies[key] - baseline[key]
    return delta


def _isolated_contribution(
    isolated_name: str,
    isolated: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    delta = _delta_vs_baseline(isolated, baseline)
    return {
        "configuration": isolated_name,
        "delta_vs_baseline": delta,
        "primary_lever": isolated_name.split("_")[0],  # e1 / e2 / e3
    }


def run_e5_ablation(
    trace: E4WorkloadTrace,
    presets: Optional[List[E5AblationPreset]] = None,
) -> E5AblationResult:
    """Run the standard E5 ablation matrix on a decoder workload trace."""
    preset_list = list(presets or DEFAULT_PRESETS)

    rows: List[E5AblationRow] = []
    baseline_proxies: Optional[Dict[str, Any]] = None

    for preset in preset_list:
        proxies = analyze_layer_proxies(trace, preset.switches)
        slice_ = _proxy_slice(proxies)
        if preset.kind == "baseline":
            baseline_proxies = slice_
        delta = (
            _delta_vs_baseline(slice_, baseline_proxies)
            if baseline_proxies is not None
            else {}
        )
        rows.append(
            E5AblationRow(
                name=preset.name,
                kind=preset.kind,
                switches=preset.switches.to_dict(),
                proxies=slice_,
                delta_vs_baseline=delta,
            )
        )

    assert baseline_proxies is not None
    by_name = {row.name: row for row in rows}

    isolated = {
        "e1_block_opt": _isolated_contribution(
            "e1_block_opt_only",
            by_name["e1_block_opt_only"].proxies,
            baseline_proxies,
        ),
        "e2_prefix_cache": _isolated_contribution(
            "e2_prefix_only",
            by_name["e2_prefix_only"].proxies,
            baseline_proxies,
        ),
        "e3_gpu_kv": _isolated_contribution(
            "e3_gpu_kv_only",
            by_name["e3_gpu_kv_only"].proxies,
            baseline_proxies,
        ),
    }

    cumulative_stack = [
        by_name[name].to_dict()
        for name in ("baseline", "cumulative_e1", "cumulative_e1_e2", "full")
        if name in by_name
    ]

    return E5AblationResult(
        trace=asdict(trace),
        configurations=[row.to_dict() for row in rows],
        isolated_contributions=isolated,
        cumulative_stack=cumulative_stack,
    )


def run_e5_multi_bucket_ablation(benchmarks_dir: Path) -> Dict[str, Any]:
    """Run E5 ablation matrix for all S1/S2/S3 decoder workload buckets."""
    from llmir.benchmark.decoder_workload_buckets import list_decoder_workload_buckets

    buckets: List[Dict[str, Any]] = []
    for bucket in list_decoder_workload_buckets():
        sim_path = bucket.artifact_path(benchmarks_dir)
        trace = trace_from_sim_json(sim_path)
        result = run_e5_ablation(trace)
        buckets.append(
            {
                "bucket_id": bucket.bucket_id,
                "bucket_label": bucket.label,
                "sim_json": str(sim_path),
                "ablation": result.to_dict(),
            }
        )
    return {
        "experiment": "E5",
        "mode": "multi_bucket_ablation",
        "benchmarks_dir": str(benchmarks_dir),
        "buckets": buckets,
    }
