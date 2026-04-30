#!/usr/bin/env python3
"""Compare CPU inference throughput between LLMIR serving and vLLM.

The vLLM path is optional and only runs when vLLM is installed with CPU support.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llmir import LLMEngine, SamplingParams  # noqa: E402
from llmir.serving.config import BackendType  # noqa: E402


@dataclass
class BenchmarkResult:
    """A single benchmark result row."""

    engine: str
    model: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float
    throughput_tokens_s: float
    latency_ms_per_token: float
    note: str = ""


def build_prompts(batch_size: int, prompt_tokens: int) -> List[str]:
    """Build deterministic prompts with approximately prompt_tokens words."""
    token = "hello"
    prompt = " ".join([token] * prompt_tokens)
    return [prompt for _ in range(batch_size)]


def time_call(fn: Callable[[], int]) -> tuple[float, int]:
    """Time a callable returning the number of generated tokens."""
    start = time.perf_counter()
    generated_tokens = fn()
    elapsed_s = time.perf_counter() - start
    return elapsed_s, generated_tokens


def make_result(
    engine: str,
    model: str,
    batch_size: int,
    prompt_tokens: int,
    generated_tokens: int,
    elapsed_s: float,
    note: str = "",
) -> BenchmarkResult:
    """Build a result with derived metrics."""
    throughput = generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    latency = elapsed_s * 1000 / generated_tokens if generated_tokens > 0 else 0.0
    return BenchmarkResult(
        engine=engine,
        model=model,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        elapsed_s=elapsed_s,
        throughput_tokens_s=throughput,
        latency_ms_per_token=latency,
        note=note,
    )


def run_llmir_cpu(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> BenchmarkResult:
    """Benchmark the LLMIR CPU serving path."""
    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine(model_path=model)

    for _ in range(warmup):
        engine.generate(prompts, params, use_tqdm=False)

    elapsed_s, generated_tokens = time_call(
        lambda: sum(
            len(output.outputs[0].token_ids)
            for output in engine.generate(prompts, params, use_tqdm=False)
        )
    )
    engine.shutdown()

    return make_result(
        "llmir",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note="LLMIR serving/token path on CPU",
    )


def run_vllm_cpu(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> Optional[BenchmarkResult]:
    """Benchmark vLLM CPU inference when vLLM is available."""
    try:
        from vllm import LLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError:
        return None

    prompts = build_prompts(batch_size, prompt_tokens)
    sampling_params = VLLMSamplingParams(max_tokens=max_tokens, temperature=0.0)
    llm = LLM(
        model=model,
        dtype="float32",
        trust_remote_code=True,
        enforce_eager=True,
    )

    for _ in range(warmup):
        llm.generate(prompts[:1], sampling_params)

    elapsed_s, generated_tokens = time_call(
        lambda: sum(
            len(output.outputs[0].token_ids)
            for output in llm.generate(prompts, sampling_params)
        )
    )

    return make_result(
        "vllm",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note="vLLM CPU baseline",
    )


def run_llmir_vllm_backend(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> Optional[BenchmarkResult]:
    """Benchmark LLMIR's ``LLMEngine`` driving the optional vLLM backend.

    Returns ``None`` when vLLM is not installed so the rest of the comparison
    can still run on CPU-only environments.
    """
    try:
        import vllm  # noqa: F401
    except ImportError:
        return None

    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine.from_pretrained(
        model,
        backend=BackendType.VLLM,
        dtype="float32",
        trust_remote_code=True,
    )

    for _ in range(warmup):
        engine.generate(prompts[:1], params, use_tqdm=False)

    elapsed_s, generated_tokens = time_call(
        lambda: sum(
            len(output.outputs[0].token_ids)
            for output in engine.generate(prompts, params, use_tqdm=False)
        )
    )
    engine.shutdown()

    return make_result(
        "llmir+vllm",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note="LLMIR LLMEngine with vLLM backend",
    )


def print_results(results: List[BenchmarkResult]) -> None:
    """Print a compact comparison table."""
    print(
        f"{'Engine':<8} {'Batch':>5} {'Prompt':>6} {'Gen':>6} "
        f"{'Time(s)':>9} {'Tok/s':>12} {'ms/tok':>10}  Note"
    )
    print("-" * 86)
    for result in results:
        print(
            f"{result.engine:<8} {result.batch_size:>5} "
            f"{result.prompt_tokens:>6} {result.generated_tokens:>6} "
            f"{result.elapsed_s:>9.4f} {result.throughput_tokens_s:>12.2f} "
            f"{result.latency_ms_per_token:>10.3f}  {result.note}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="CPU inference benchmark comparing LLMIR with vLLM"
    )
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument(
        "--skip-llmir-vllm-backend",
        action="store_true",
        help="Skip the LLMIR LLMEngine path that drives the vLLM backend",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    """Run the CPU comparison benchmark."""
    args = parse_args()
    results = [
        run_llmir_cpu(
            args.model,
            args.batch_size,
            args.prompt_tokens,
            args.max_tokens,
            args.warmup,
        )
    ]

    if not args.skip_vllm:
        vllm_result = run_vllm_cpu(
            args.model,
            args.batch_size,
            args.prompt_tokens,
            args.max_tokens,
            args.warmup,
        )
        if vllm_result is None:
            print("vLLM is not installed; skipping vLLM CPU baseline.")
        else:
            results.append(vllm_result)

    if not args.skip_llmir_vllm_backend:
        llmir_vllm_result = run_llmir_vllm_backend(
            args.model,
            args.batch_size,
            args.prompt_tokens,
            args.max_tokens,
            args.warmup,
        )
        if llmir_vllm_result is None:
            print("vLLM is not installed; skipping LLMIR+vLLM backend path.")
        else:
            results.append(llmir_vllm_result)

    print_results(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([asdict(result) for result in results], f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
