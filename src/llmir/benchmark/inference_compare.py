"""Multi-backend inference comparison (HF, vLLM, LLMIR paths)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

from llmir import LLMEngine, SamplingParams
from llmir.serving.config import BackendType


@dataclass
class BenchmarkResult:
    """A single backend benchmark row."""

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
    token = "hello"
    prompt = " ".join([token] * prompt_tokens)
    return [prompt for _ in range(batch_size)]


def time_call(fn: Callable[[], int]) -> tuple[float, int]:
    start = time.perf_counter()
    generated_tokens = fn()
    return time.perf_counter() - start, generated_tokens


def make_result(
    engine: str,
    model: str,
    batch_size: int,
    prompt_tokens: int,
    generated_tokens: int,
    elapsed_s: float,
    note: str = "",
) -> BenchmarkResult:
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


def run_llmir_smoke(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> BenchmarkResult:
    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine(model_path=model, backend="llmir")
    for _ in range(warmup):
        engine.generate(prompts, params, use_tqdm=False)
    elapsed_s, generated_tokens = time_call(
        lambda: sum(
            len(o.outputs[0].token_ids)
            for o in engine.generate(prompts, params, use_tqdm=False)
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
        note="placeholder smoke path",
    )


def run_hf_transformers(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> Optional[BenchmarkResult]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    torch_model.eval()
    prompts = build_prompts(batch_size, prompt_tokens)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    for _ in range(warmup):
        with torch.no_grad():
            torch_model.generate(
                **inputs,
                max_new_tokens=min(4, max_tokens),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    def _run() -> int:
        with torch.no_grad():
            out = torch_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_per = out.shape[1] - inputs["input_ids"].shape[1]
        return batch_size * gen_per

    elapsed_s, generated_tokens = time_call(_run)
    del torch_model
    return make_result(
        "hf",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note="HuggingFace generate() baseline",
    )


def run_vllm(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> Optional[BenchmarkResult]:
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
            len(o.outputs[0].token_ids) for o in llm.generate(prompts, sampling_params)
        )
    )
    return make_result(
        "vllm",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note="vLLM baseline",
    )


def run_llmir_vllm_backend(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> Optional[BenchmarkResult]:
    try:
        import vllm  # noqa: F401
    except ImportError:
        return None

    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine.from_pretrained(
        model, backend=BackendType.VLLM, dtype="float32", trust_remote_code=True
    )
    for _ in range(warmup):
        engine.generate(prompts[:1], params, use_tqdm=False)
    elapsed_s, generated_tokens = time_call(
        lambda: sum(
            len(o.outputs[0].token_ids)
            for o in engine.generate(prompts, params, use_tqdm=False)
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
        note="LLMIR engine forwarding to vLLM",
    )


def run_llmir_paged(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
) -> Optional[BenchmarkResult]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return None

    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine.from_pretrained(
        model, backend=BackendType.LLMIR_PAGED, dtype="float32", trust_remote_code=True
    )
    for _ in range(warmup):
        engine.generate(prompts[:1], params, use_tqdm=False)
    elapsed_s, generated_tokens = time_call(
        lambda: sum(
            len(o.outputs[0].token_ids)
            for o in engine.generate(prompts, params, use_tqdm=False)
        )
    )
    engine.shutdown()
    return make_result(
        "llmir-paged",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note="LLMIR PagedKVCache on critical path",
    )


_BACKEND_RUNNERS: Dict[str, Callable[..., Optional[BenchmarkResult]]] = {
    "llmir": run_llmir_smoke,
    "hf": run_hf_transformers,
    "huggingface": run_hf_transformers,
    "transformers": run_hf_transformers,
    "vllm": run_vllm,
    "llmir+vllm": run_llmir_vllm_backend,
    "llmir_vllm": run_llmir_vllm_backend,
    "llmir-paged": run_llmir_paged,
    "llmir_paged": run_llmir_paged,
}


def normalize_backend_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def run_inference_compare(
    model: str,
    backends: List[str],
    *,
    batch_size: int = 1,
    prompt_tokens: int = 16,
    max_tokens: int = 16,
    warmup: int = 1,
) -> List[BenchmarkResult]:
    """Run one or more backends; skip unavailable ones with a warning."""
    results: List[BenchmarkResult] = []
    aliases = {
        "huggingface": "hf",
        "transformers": "hf",
        "llmir-vllm": "llmir+vllm",
        "llmir-paged": "llmir-paged",
    }
    for raw in backends:
        key = normalize_backend_name(raw)
        key = aliases.get(key, key)
        runner = _BACKEND_RUNNERS.get(key)
        if runner is None:
            print(f"Unknown backend {raw!r}; known: {sorted(set(_BACKEND_RUNNERS))}")
            continue
        row = runner(model, batch_size, prompt_tokens, max_tokens, warmup)
        if row is None:
            print(f"Skipping {raw}: dependencies not installed")
        else:
            results.append(row)
    return results


def print_inference_results(results: List[BenchmarkResult]) -> None:
    print(
        f"{'Engine':<12} {'Batch':>5} {'Prompt':>6} {'Gen':>6} "
        f"{'Time(s)':>9} {'Tok/s':>12} {'ms/tok':>10}  Note"
    )
    print("-" * 90)
    for result in results:
        print(
            f"{result.engine:<12} {result.batch_size:>5} "
            f"{result.prompt_tokens:>6} {result.generated_tokens:>6} "
            f"{result.elapsed_s:>9.4f} {result.throughput_tokens_s:>12.2f} "
            f"{result.latency_ms_per_token:>10.3f}  {result.note}"
        )


def results_to_json(results: List[BenchmarkResult]) -> List[dict]:
    return [asdict(r) for r in results]
