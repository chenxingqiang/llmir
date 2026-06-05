"""Multi-backend inference comparison (HF, vLLM, LLMIR paths)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from llmir import LLMEngine, SamplingParams
from llmir.benchmark.device import (
    DtypeChoice,
    DeviceChoice,
    InferenceDeviceConfig,
    cuda_available,
    hf_device_map,
    resolve_inference_device,
    resolve_torch_dtype,
    torch_dtype_from_string,
    vllm_dtype_string,
)
from llmir.serving.config import BackendType


@dataclass
class InferenceCompareConfig:
    """Runtime options for :func:`run_inference_compare`."""

    device: DeviceChoice = "auto"
    dtype: DtypeChoice = "auto"
    resolved: InferenceDeviceConfig = field(init=False)

    def __post_init__(self) -> None:
        base = resolve_inference_device(self.device)
        dtype_str = resolve_torch_dtype(self.dtype, base)
        object.__setattr__(
            self,
            "resolved",
            InferenceDeviceConfig(
                device=base.device,
                torch_dtype=dtype_str,
                note=base.note,
            ),
        )


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
    device: str = ""
    dtype: str = ""


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
    *,
    device: str = "",
    dtype: str = "",
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
        device=device,
        dtype=dtype,
    )


def run_llmir_smoke(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
    compare_config: Optional[InferenceCompareConfig] = None,
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
    compare_config: Optional[InferenceCompareConfig] = None,
) -> Optional[BenchmarkResult]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None

    cfg = compare_config or InferenceCompareConfig()
    torch_dtype = torch_dtype_from_string(cfg.resolved.torch_dtype) or torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        device_map=hf_device_map(cfg.resolved.device),
        trust_remote_code=True,
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
    note = f"HuggingFace generate() ({cfg.resolved.note})"
    return make_result(
        "hf",
        model,
        batch_size,
        prompt_tokens,
        generated_tokens,
        elapsed_s,
        note=note,
        device=cfg.resolved.device,
        dtype=cfg.resolved.torch_dtype,
    )


def run_vllm(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
    compare_config: Optional[InferenceCompareConfig] = None,
) -> Optional[BenchmarkResult]:
    try:
        from vllm import LLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError:
        return None

    cfg = compare_config or InferenceCompareConfig()
    prompts = build_prompts(batch_size, prompt_tokens)
    sampling_params = VLLMSamplingParams(max_tokens=max_tokens, temperature=0.0)
    llm_kwargs: Dict[str, object] = {
        "model": model,
        "dtype": vllm_dtype_string(cfg.resolved.torch_dtype),
        "trust_remote_code": True,
        "enforce_eager": True,
    }
    if cfg.resolved.device == "cpu":
        llm_kwargs["device"] = "cpu"
    llm = LLM(**llm_kwargs)
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
        note=f"vLLM baseline ({cfg.resolved.note})",
        device=cfg.resolved.device,
        dtype=cfg.resolved.torch_dtype,
    )


def run_llmir_vllm_backend(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
    compare_config: Optional[InferenceCompareConfig] = None,
) -> Optional[BenchmarkResult]:
    try:
        import vllm  # noqa: F401
    except ImportError:
        return None

    cfg = compare_config or InferenceCompareConfig()
    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine.from_pretrained(
        model,
        backend=BackendType.VLLM,
        dtype=cfg.resolved.torch_dtype,
        trust_remote_code=True,
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
        note=f"LLMIR→vLLM ({cfg.resolved.note})",
        device=cfg.resolved.device,
        dtype=cfg.resolved.torch_dtype,
    )


def run_llmir_paged(
    model: str,
    batch_size: int,
    prompt_tokens: int,
    max_tokens: int,
    warmup: int,
    compare_config: Optional[InferenceCompareConfig] = None,
) -> Optional[BenchmarkResult]:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return None

    cfg = compare_config or InferenceCompareConfig()
    prompts = build_prompts(batch_size, prompt_tokens)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine.from_pretrained(
        model,
        backend=BackendType.LLMIR_PAGED,
        dtype=cfg.resolved.torch_dtype,
        trust_remote_code=True,
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
        note=f"LLMIR PagedKV ({cfg.resolved.note})",
        device=cfg.resolved.device,
        dtype=cfg.resolved.torch_dtype,
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
    device: DeviceChoice = "auto",
    dtype: DtypeChoice = "auto",
) -> List[BenchmarkResult]:
    """Run one or more backends; skip unavailable ones with a warning."""
    compare_config = InferenceCompareConfig(device=device, dtype=dtype)
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
        row = runner(
            model,
            batch_size,
            prompt_tokens,
            max_tokens,
            warmup,
            compare_config,
        )
        if row is None:
            print(f"Skipping {raw}: dependencies not installed")
        else:
            results.append(row)
    return results


def print_inference_results(results: List[BenchmarkResult]) -> None:
    try:
        from llmir.runtime.cuda_probe import summarize_cuda_stack

        stack = summarize_cuda_stack()
        print(
            f"CUDA stack: torch={stack['torch_cuda']} "
            f"native_built={stack['native_cuda_built']} "
            f"native_runtime={stack['native_cuda_runtime']} "
            f"devices={stack['device_count']}"
        )
    except ImportError:
        pass
    if results:
        print(f"Device: {results[0].device or 'n/a'}  dtype: {results[0].dtype or 'n/a'}")
        if cuda_available():
            print("(CUDA available for compare)")
    print(
        f"{'Engine':<12} {'Batch':>5} {'Prompt':>6} {'Gen':>6} "
        f"{'Time(s)':>9} {'Tok/s':>12} {'ms/tok':>10}  Note"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result.engine:<12} {result.batch_size:>5} "
            f"{result.prompt_tokens:>6} {result.generated_tokens:>6} "
            f"{result.elapsed_s:>9.4f} {result.throughput_tokens_s:>12.2f} "
            f"{result.latency_ms_per_token:>10.3f}  {result.note}"
        )


def results_to_json(results: List[BenchmarkResult]) -> List[dict]:
    return [asdict(r) for r in results]
