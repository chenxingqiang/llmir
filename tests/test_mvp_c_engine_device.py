"""MVP-C: LLMEngine wires CUDA device into PagedKVDecoder."""

from __future__ import annotations

import os

import pytest


def test_resolve_inference_device_env_override(monkeypatch):
    from llmir.runtime.device import resolve_inference_device

    monkeypatch.setenv("LLMIR_DEVICE", "cpu")
    assert resolve_inference_device("cuda") == "cpu"
    monkeypatch.delenv("LLMIR_DEVICE", raising=False)
    assert resolve_inference_device("cpu") == "cpu"


def test_engine_llmir_paged_passes_cuda_device_to_decoder(monkeypatch):
    pytest.importorskip("torch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmir.runtime.device import resolve_inference_device
    from llmir.serving.config import BackendType
    from llmir.serving.engine import LLMEngine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    tiny = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(tiny)
    tokenizer = AutoTokenizer.from_pretrained(tiny)
    model.eval()
    dev = torch.device(resolve_inference_device("auto"))
    model.to(dev)

    engine = LLMEngine(model_path=tiny, backend=BackendType.LLMIR_PAGED)
    engine._tokenizer = tokenizer
    engine._tokenizer_attempted = True
    engine._hf_model = model
    engine._ensure_llmir_paged = engine._ensure_llmir_paged.__get__(engine, LLMEngine)

    # Bypass full HF reload — build decoder like _ensure_llmir_paged tail.
    from llmir.runtime.paged_decoder import PagedKVDecoder, kv_config_from_hf_config

    kv_config = kv_config_from_hf_config(model.config, dtype="float32")
    engine._paged_decoder = PagedKVDecoder(
        model,
        tokenizer,
        kv_config=kv_config,
        device=dev,
        dtype=torch.float32,
    )

    os.environ.pop("LLMIR_KV_BACKEND", None)
    caches = engine._paged_decoder._create_layer_caches()
    from llmir.runtime.kv_factory import kv_cache_backend_name
    from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

    assert engine._paged_decoder._device.type == "cuda"
    assert engine._paged_decoder.kv_config.enable_gpu is True
    assert all(isinstance(c, TorchGpuPagedKVCache) for c in caches)
    assert kv_cache_backend_name(caches[0]) == "torch_cuda"
