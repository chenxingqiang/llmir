"""Device and dtype helpers for inference benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

DeviceChoice = Literal["auto", "cpu", "cuda"]
DtypeChoice = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class InferenceDeviceConfig:
    """Resolved runtime placement for compare benchmarks."""

    device: str
    torch_dtype: str
    note: str = ""


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_inference_device(
    choice: DeviceChoice = "auto",
) -> InferenceDeviceConfig:
    """
    Pick device and a sensible default dtype for E2E compare runs.

    ``auto`` prefers CUDA when available.
    """
    if choice == "cpu":
        return InferenceDeviceConfig(device="cpu", torch_dtype="float32", note="cpu")
    if choice == "cuda":
        if not cuda_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return InferenceDeviceConfig(
            device="cuda",
            torch_dtype="float16",
            note="cuda",
        )
    if cuda_available():
        return InferenceDeviceConfig(
            device="cuda",
            torch_dtype="float16",
            note="auto→cuda",
        )
    return InferenceDeviceConfig(device="cpu", torch_dtype="float32", note="auto→cpu")


def resolve_torch_dtype(
    choice: DtypeChoice,
    device_config: InferenceDeviceConfig,
) -> str:
    """Return torch dtype string for model loading."""
    if choice == "auto":
        return device_config.torch_dtype
    return choice


def vllm_dtype_string(torch_dtype: str) -> str:
    """Map torch dtype names to vLLM ``dtype`` argument."""
    return torch_dtype.replace("torch.", "")


def hf_device_map(device: str) -> Optional[str]:
    """HuggingFace ``device_map`` value (legacy; prefer :func:`hf_from_pretrained_kwargs`)."""
    if device == "cuda":
        return None
    return "cpu"


def torch_dtype_from_string(name: str) -> Optional["object"]:
    try:
        import torch
    except ImportError:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(name.lower())
