"""CUDA capability probes for LLMIR native runtime and PyTorch."""

from __future__ import annotations

from typing import Any, Dict

from llmir.runtime.native_bridge import native_library_available


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def native_cuda_built() -> bool:
    """True if ``libMLIRLLMRuntime`` was compiled with ``LLMIR_ENABLE_CUDA``."""
    if not native_library_available():
        return False
    try:
        from llmir.runtime.native_bridge import _load_library

        lib = _load_library()
        if not hasattr(lib, "llmir_has_cuda_support"):
            return False
        return bool(lib.llmir_has_cuda_support())
    except (RuntimeError, OSError, AttributeError):
        return False


def native_cuda_runtime_available() -> bool:
    """True when native CUDA kernels can run (built + visible device)."""
    if not native_library_available():
        return False
    try:
        from llmir.runtime.native_bridge import _load_library

        lib = _load_library()
        if not hasattr(lib, "llmir_cuda_runtime_available"):
            return native_cuda_built() and torch_cuda_available()
        return bool(lib.llmir_cuda_runtime_available())
    except (RuntimeError, OSError, AttributeError):
        return False


def cuda_device_count() -> int:
    if native_library_available():
        try:
            from llmir.runtime.native_bridge import _load_library

            lib = _load_library()
            if hasattr(lib, "llmir_cuda_device_count"):
                return int(lib.llmir_cuda_device_count())
        except (RuntimeError, OSError, AttributeError):
            pass
    if torch_cuda_available():
        import torch

        return int(torch.cuda.device_count())
    return 0


def summarize_cuda_stack() -> Dict[str, Any]:
    """JSON-serializable snapshot for benchmarks and CI logs."""
    return {
        "torch_cuda": torch_cuda_available(),
        "native_cuda_built": native_cuda_built(),
        "native_cuda_runtime": native_cuda_runtime_available(),
        "device_count": cuda_device_count(),
    }
