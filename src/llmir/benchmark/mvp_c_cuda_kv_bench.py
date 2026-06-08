"""
MVP-C: compare NumPy vs torch-GPU KV backends for llmir_paged decode.

Measures end-to-end ``PagedKVDecoder`` latency with ``LLMIR_KV_BACKEND`` set to
``numpy`` vs ``torch_cuda`` (and optionally ``native`` when the C++ library is
built with CUDA).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmir.runtime.cuda_probe import summarize_cuda_stack


@dataclass
class MVPCudaKVBenchConfig:
    model: str = "gpt2"
    prompt: str = "The quick brown fox jumps over the lazy dog. " * 8
    max_new_tokens: int = 16
    warmup: int = 1
    backends: List[str] = field(default_factory=lambda: ["numpy", "torch_cuda"])
    device: Optional[str] = None


@dataclass
class MVPCudaKVBenchResult:
    backend: str
    total_seconds: float
    tokens_per_second: float
    prompt_tokens: int
    generated_tokens: int
    kv_backend_label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "kv_backend_label": self.kv_backend_label,
            "total_seconds": self.total_seconds,
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
        }


def _run_one_backend(
    cfg: MVPCudaKVBenchConfig,
    kv_backend: str,
) -> MVPCudaKVBenchResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmir.integration.hf_load import hf_from_pretrained_kwargs, materialize_hf_causal_lm
    from llmir.runtime.kv_factory import kv_cache_backend_name
    from llmir.runtime.paged_decoder import PagedKVDecoder

    prev = os.environ.get("LLMIR_KV_BACKEND")
    os.environ["LLMIR_KV_BACKEND"] = kv_backend
    try:
        from llmir.runtime.device import resolve_inference_device

        device = resolve_inference_device(cfg.device or "auto")

        dtype = torch.float16 if device == "cuda" else torch.float32
        load_kw = hf_from_pretrained_kwargs(device=device, torch_dtype=dtype)
        model = materialize_hf_causal_lm(
            AutoModelForCausalLM.from_pretrained(cfg.model, **load_kw)
        )
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
        decoder = PagedKVDecoder(
            model,
            tokenizer,
            enable_prefix_cache=False,
        )
        kv_label = kv_cache_backend_name(decoder._create_layer_caches()[0])

        for _ in range(max(cfg.warmup, 0)):
            decoder.decode([cfg.prompt], max_new_tokens=1)

        t0 = time.perf_counter()
        results = decoder.decode([cfg.prompt], max_new_tokens=cfg.max_new_tokens)
        elapsed = time.perf_counter() - t0

        out = results[0]
        gen = len(out.generated_token_ids)
        tps = gen / elapsed if elapsed > 0 else 0.0
        return MVPCudaKVBenchResult(
            backend=kv_backend,
            kv_backend_label=kv_label,
            total_seconds=elapsed,
            tokens_per_second=tps,
            prompt_tokens=len(out.prompt_token_ids),
            generated_tokens=gen,
        )
    finally:
        if prev is None:
            os.environ.pop("LLMIR_KV_BACKEND", None)
        else:
            os.environ["LLMIR_KV_BACKEND"] = prev


def run_mvp_c_cuda_kv_benchmark(
    cfg: Optional[MVPCudaKVBenchConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MVPCudaKVBenchConfig()
    rows: List[Dict[str, Any]] = []
    for backend in cfg.backends:
        try:
            rows.append(_run_one_backend(cfg, backend).to_dict())
        except Exception as exc:  # pragma: no cover - optional native/cuda
            rows.append(
                {
                    "backend": backend,
                    "kv_backend_label": "error",
                    "total_seconds": 0.0,
                    "tokens_per_second": 0.0,
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "error": str(exc),
                }
            )

    baseline_tps = next(
        (
            r["tokens_per_second"]
            for r in rows
            if r.get("backend") == "numpy" and r.get("tokens_per_second")
        ),
        None,
    )
    payload: Dict[str, Any] = {
        "mode": "mvp_c_cuda_kv",
        "model": cfg.model,
        "cuda_stack": summarize_cuda_stack(),
        "results": [],
    }
    for row in rows:
        d = dict(row)
        tps = d.get("tokens_per_second") or 0.0
        if baseline_tps and tps and d.get("backend") != "numpy":
            d["speedup_vs_numpy_tps"] = tps / baseline_tps
        payload["results"].append(d)
    return payload


def print_mvp_c_results(payload: Dict[str, Any]) -> None:
    print(f"Model: {payload.get('model')}")
    stack = payload.get("cuda_stack", {})
    print(
        f"CUDA: torch={stack.get('torch_cuda')} "
        f"native_built={stack.get('native_cuda_built')} "
        f"native_runtime={stack.get('native_cuda_runtime')} "
        f"devices={stack.get('device_count')}"
    )
    print("-" * 60)
    for row in payload.get("results", []):
        label = row.get("kv_backend_label", row.get("backend"))
        tps = row.get("tokens_per_second", 0.0)
        sec = row.get("total_seconds", 0.0)
        speedup = row.get("speedup_vs_numpy_tps")
        line = f"{row.get('backend'):12} ({label:10})  {tps:8.2f} tok/s  ({sec:.3f}s)"
        if speedup is not None:
            line += f"  speedup={speedup:.2f}x"
        if "error" in row:
            line += f"  ERROR: {row['error']}"
        print(line)
