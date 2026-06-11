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
    """CPU-safe: verify cuda device string reaches PagedKVDecoder and kv factory."""
    pytest.importorskip("torch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmir.runtime.device import resolve_inference_device
    from llmir.runtime.kv_factory import create_paged_kv_cache
    from llmir.runtime.paged_decoder import PagedKVDecoder, kv_config_from_hf_config
    from llmir.serving.config import BackendType
    from llmir.serving.engine import LLMEngine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    tiny = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(tiny)
    tokenizer = AutoTokenizer.from_pretrained(tiny)
    model.eval()

    assert resolve_inference_device("auto") == "cuda"
    dev = torch.device("cuda")

    engine = LLMEngine(model_path=tiny, backend=BackendType.LLMIR_PAGED)
    engine._tokenizer = tokenizer
    engine._tokenizer_attempted = True
    engine._hf_model = model

    kv_config = kv_config_from_hf_config(model.config, dtype="float32")
    engine._paged_decoder = PagedKVDecoder(
        model,
        tokenizer,
        kv_config=kv_config,
        device=dev,
        dtype=torch.float32,
    )

    assert engine._paged_decoder._device.type == "cuda"
    assert engine._paged_decoder.kv_config.enable_gpu is True

    captured_devices: list[str | None] = []
    real_create = create_paged_kv_cache

    def record_device(config, *, prefer_native=None, device=None):
        captured_devices.append(device)
        return real_create(config, prefer_native=prefer_native, device="cpu")

    monkeypatch.setattr(
        "llmir.runtime.paged_decoder.create_paged_kv_cache", record_device
    )
    os.environ.pop("LLMIR_KV_BACKEND", None)
    caches = engine._paged_decoder._create_layer_caches()

    assert captured_devices
    assert all(d == "cuda" for d in captured_devices)
    assert len(caches) == engine._paged_decoder.kv_config.num_layers


@pytest.mark.skipif(
    not pytest.importorskip("torch").cuda.is_available(),
    reason="needs CUDA hardware",
)
def test_engine_llmir_paged_creates_torch_gpu_caches_on_cuda():
    """GPU integration: TorchGpuPagedKVCache when model and decoder run on CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmir.runtime.kv_factory import kv_cache_backend_name
    from llmir.runtime.paged_decoder import PagedKVDecoder, kv_config_from_hf_config
    from llmir.runtime.torch_gpu_kv_cache import TorchGpuPagedKVCache

    tiny = "sshleifer/tiny-gpt2"
    model = AutoModelForCausalLM.from_pretrained(tiny)
    tokenizer = AutoTokenizer.from_pretrained(tiny)
    model.eval()
    dev = torch.device("cuda")
    model.to(dev)

    kv_config = kv_config_from_hf_config(model.config, dtype="float32")
    decoder = PagedKVDecoder(
        model,
        tokenizer,
        kv_config=kv_config,
        device=dev,
        dtype=torch.float32,
    )

    os.environ.pop("LLMIR_KV_BACKEND", None)
    caches = decoder._create_layer_caches()

    assert decoder._device.type == "cuda"
    assert decoder.kv_config.enable_gpu is True
    assert all(isinstance(c, TorchGpuPagedKVCache) for c in caches)
    assert kv_cache_backend_name(caches[0]) == "torch_cuda"
