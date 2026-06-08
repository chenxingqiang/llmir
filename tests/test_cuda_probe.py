"""CUDA probe helpers (no GPU required)."""

from llmir.runtime.cuda_probe import (
    cuda_device_count,
    native_cuda_built,
    summarize_cuda_stack,
    torch_cuda_available,
)


def test_summarize_cuda_stack_keys():
    info = summarize_cuda_stack()
    assert set(info.keys()) == {
        "torch_cuda",
        "native_cuda_built",
        "native_cuda_runtime",
        "device_count",
    }
    assert isinstance(info["torch_cuda"], bool)
    assert isinstance(info["device_count"], int)
    assert info["device_count"] >= 0


def test_native_cuda_built_without_lib():
    if not __import__("llmir.runtime.native_bridge", fromlist=["native_library_available"]).native_library_available():
        assert native_cuda_built() is False


def test_torch_cuda_is_bool():
    assert isinstance(torch_cuda_available(), bool)
    assert isinstance(cuda_device_count(), int)
