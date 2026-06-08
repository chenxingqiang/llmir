"""ShareGPT-style prefix workload benchmarks (MVP-B)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llmir.benchmark.prefix_cache_bench import PrefixBenchResult, bench_prefix_kv_reuse


@dataclass(frozen=True)
class ShareGPTPrefixBenchConfig:
    """
    Synthetic ShareGPT-like load: one long shared prefix + many user variants.

    Paper §5.1 uses ShareGPT with mixed sequence lengths; MVP-B fixes the
    high-prefix-overlap regime (system prompt reuse).
    """

    system_prompt_tokens: int = 128
    num_requests: int = 32
    user_suffix_tokens: int = 8
    max_new_tokens: int = 4
    model: str = "gpt2"
    device: str = "auto"
    warmup_requests: int = 0


@dataclass
class ShareGPTRequestMetrics:
    """Per-request metrics from ``llmir_paged`` decode."""

    request_index: int
    elapsed_s: float
    prefix_hit_tokens: int = 0
    prefill_tokens_computed: int = 0
    generated_tokens: int = 0


@dataclass
class ShareGPTPrefixBenchResult:
    """Aggregated MVP-B benchmark row."""

    scenario: str
    num_requests: int
    system_prompt_tokens: int
    user_suffix_tokens: int
    total_elapsed_s: float
    avg_latency_ms: float
    ttft_proxy_ms: float
    prefix_hit_tokens_total: int
    prefill_tokens_computed_total: int
    generated_tokens_total: int
    throughput_tok_s: float
    speedup_vs_baseline: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_sharegpt_prompts(
    cfg: ShareGPTPrefixBenchConfig,
    *,
    token_word: str = "hello",
) -> tuple[str, List[str]]:
    """
    Build a long system prefix string and ``num_requests`` full prompts.

    Token count is approximate (whitespace-separated words); when a tokenizer
    is available, callers should re-measure for exact counts.
    """
    system = " ".join([token_word] * cfg.system_prompt_tokens)
    prompts: List[str] = []
    for i in range(cfg.num_requests):
        suffix = " ".join([f"user{i}", token_word] * max(1, cfg.user_suffix_tokens // 2))
        prompts.append(f"{system} {suffix}".strip())
    return system, prompts


def _aggregate_metrics(
    scenario: str,
    cfg: ShareGPTPrefixBenchConfig,
    rows: Sequence[ShareGPTRequestMetrics],
    *,
    speedup: float = 1.0,
    details: Optional[Dict[str, Any]] = None,
) -> ShareGPTPrefixBenchResult:
    total_s = sum(r.elapsed_s for r in rows)
    gen_total = sum(r.generated_tokens for r in rows)
    hit_total = sum(r.prefix_hit_tokens for r in rows)
    prefill_total = sum(r.prefill_tokens_computed for r in rows)
    ttft = rows[0].elapsed_s * 1000.0 if rows else 0.0
    avg_ms = (total_s / len(rows) * 1000.0) if rows else 0.0
    throughput = gen_total / total_s if total_s > 0 else 0.0
    return ShareGPTPrefixBenchResult(
        scenario=scenario,
        num_requests=cfg.num_requests,
        system_prompt_tokens=cfg.system_prompt_tokens,
        user_suffix_tokens=cfg.user_suffix_tokens,
        total_elapsed_s=total_s,
        avg_latency_ms=avg_ms,
        ttft_proxy_ms=ttft,
        prefix_hit_tokens_total=hit_total,
        prefill_tokens_computed_total=prefill_total,
        generated_tokens_total=gen_total,
        throughput_tok_s=throughput,
        speedup_vs_baseline=speedup,
        details=details or {},
    )


def bench_sharegpt_kv_simulation(
    cfg: Optional[ShareGPTPrefixBenchConfig] = None,
) -> List[ShareGPTPrefixBenchResult]:
    """
    CI-friendly KV-layer simulation (no HuggingFace).

    Maps ShareGPT parameters onto :func:`bench_prefix_kv_reuse`.
    """
    cfg = cfg or ShareGPTPrefixBenchConfig()
    rows = bench_prefix_kv_reuse(
        num_requests=cfg.num_requests,
        prefix_len=cfg.system_prompt_tokens,
        suffix_len=cfg.user_suffix_tokens,
    )
    baseline = next(r for r in rows if r.scenario == "kv_append_baseline")
    optimized = next(r for r in rows if r.scenario == "kv_append_prefix_cached")
    speedup = optimized.speedup_vs_baseline

    def _to_sharegpt(row: PrefixBenchResult, scenario: str) -> ShareGPTPrefixBenchResult:
        return ShareGPTPrefixBenchResult(
            scenario=scenario,
            num_requests=cfg.num_requests,
            system_prompt_tokens=cfg.system_prompt_tokens,
            user_suffix_tokens=cfg.user_suffix_tokens,
            total_elapsed_s=row.elapsed_s,
            avg_latency_ms=row.elapsed_s * 1000 / max(cfg.num_requests, 1),
            ttft_proxy_ms=row.elapsed_s * 1000 / max(cfg.num_requests, 1),
            prefix_hit_tokens_total=0,
            prefill_tokens_computed_total=0,
            generated_tokens_total=int(row.throughput_ops_s * row.elapsed_s),
            throughput_tok_s=row.throughput_ops_s,
            speedup_vs_baseline=speedup if scenario != "sharegpt_kv_baseline" else 1.0,
            details=row.details or {},
        )

    return [
        _to_sharegpt(baseline, "sharegpt_kv_baseline"),
        _to_sharegpt(optimized, "sharegpt_kv_prefix_cached"),
    ]


def _resolve_device(device: str) -> str:
    from llmir.runtime.device import resolve_inference_device

    return resolve_inference_device(device)


def _disable_prefix_cache(engine: Any) -> None:
    decoder = getattr(engine, "_paged_decoder", None)
    if decoder is None:
        engine._ensure_llmir_paged()
        decoder = engine._paged_decoder
    if decoder is not None:
        decoder.enable_prefix_cache = False
        decoder._prefix_store = None


def bench_llmir_paged_sharegpt(
    cfg: Optional[ShareGPTPrefixBenchConfig] = None,
    *,
    with_warm_prefix: bool = True,
) -> List[ShareGPTRequestMetrics]:
    """Run ShareGPT workload on ``llmir_paged`` (requires ``llmir[full]``)."""
    cfg = cfg or ShareGPTPrefixBenchConfig()
    from llmir import LLMEngine, SamplingParams

    resolved = _resolve_device(cfg.device)
    dtype = "float16" if resolved == "cuda" else "float32"
    system, prompts = build_sharegpt_prompts(cfg)

    engine = LLMEngine.from_pretrained(
        cfg.model,
        backend="llmir_paged",
        dtype=dtype,
        trust_remote_code=True,
    )
    if not with_warm_prefix:
        _disable_prefix_cache(engine)

    if with_warm_prefix:
        engine._ensure_llmir_paged()
        assert engine._paged_decoder is not None
        warmed = engine._paged_decoder.warm_prefix(system)
    else:
        warmed = 0

    params = SamplingParams(max_tokens=cfg.max_new_tokens, temperature=0.0)
    metrics: List[ShareGPTRequestMetrics] = []

    for i in range(cfg.warmup_requests):
        engine.generate([prompts[i % len(prompts)]], params, use_tqdm=False)

    for i, prompt in enumerate(prompts):
        start = time.perf_counter()
        out = engine.generate([prompt], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - start
        m = out.metrics or {}
        metrics.append(
            ShareGPTRequestMetrics(
                request_index=i,
                elapsed_s=elapsed,
                prefix_hit_tokens=int(m.get("prefix_hit_tokens", 0)),
                prefill_tokens_computed=int(m.get("prefill_tokens_computed", 0)),
                generated_tokens=len(out.outputs[0].token_ids),
            )
        )

    engine.shutdown()
    return metrics


def run_sharegpt_prefix_benchmark(
    cfg: Optional[ShareGPTPrefixBenchConfig] = None,
    *,
    run_simulation: bool = True,
    run_llmir_paged: bool = True,
) -> Dict[str, Any]:
    """
    Full MVP-B suite: KV simulation + optional ``llmir_paged`` E2E.

    Returns JSON-serializable payload with baseline vs prefix-warmed rows.
    """
    cfg = cfg or ShareGPTPrefixBenchConfig()
    payload: Dict[str, Any] = {
        "mode": "sharegpt_prefix",
        "config": asdict(cfg),
        "results": [],
    }

    if run_simulation:
        payload["results"].extend(
            r.to_dict() for r in bench_sharegpt_kv_simulation(cfg)
        )

    if run_llmir_paged:
        try:
            baseline_rows = bench_llmir_paged_sharegpt(cfg, with_warm_prefix=False)
            warmed_rows = bench_llmir_paged_sharegpt(cfg, with_warm_prefix=True)
        except ImportError as exc:
            payload["llmir_paged_error"] = str(exc)
            return payload

        baseline = _aggregate_metrics(
            "sharegpt_llmir_no_prefix",
            cfg,
            baseline_rows,
            details={"prefix_cache": "disabled"},
        )
        warmed = _aggregate_metrics(
            "sharegpt_llmir_warm_prefix",
            cfg,
            warmed_rows,
            details={"prefix_cache": "warm_prefix"},
        )
        if warmed.total_elapsed_s > 0:
            warmed.speedup_vs_baseline = baseline.total_elapsed_s / warmed.total_elapsed_s
        payload["results"].extend([baseline.to_dict(), warmed.to_dict()])
        payload["per_request"] = {
            "baseline": [asdict(r) for r in baseline_rows],
            "warmed": [asdict(r) for r in warmed_rows],
        }

    return payload


def print_sharegpt_results(payload: Dict[str, Any]) -> None:
    """Human-readable summary."""
    print(f"ShareGPT prefix benchmark  requests={payload['config'].get('num_requests')}")
    print(
        f"  system≈{payload['config'].get('system_prompt_tokens')} tok  "
        f"suffix≈{payload['config'].get('user_suffix_tokens')} tok"
    )
    print(f"{'scenario':<32} {'avg_ms':>8} {'hit':>8} {'prefill':>8} {'tok/s':>10} {'speedup':>8}")
    print("-" * 80)
    for row in payload.get("results", []):
        print(
            f"{row['scenario']:<32} {row['avg_latency_ms']:>8.1f} "
            f"{row['prefix_hit_tokens_total']:>8} "
            f"{row['prefill_tokens_computed_total']:>8} "
            f"{row['throughput_tok_s']:>10.1f} "
            f"{row['speedup_vs_baseline']:>8.2f}x"
        )
