"""Device resolution for LLMIR runtime and serving."""

from __future__ import annotations

import os


def resolve_inference_device(device: str = "auto") -> str:
    """
    Pick ``cuda`` or ``cpu`` for HF + PagedKVDecoder inference.

    ``LLMIR_DEVICE`` overrides the ``device`` argument when set.
    """
    explicit = os.environ.get("LLMIR_DEVICE", "").strip().lower()
    if explicit:
        return explicit
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
