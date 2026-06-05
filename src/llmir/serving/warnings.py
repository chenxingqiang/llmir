"""User-visible warnings for serving backend selection."""

from __future__ import annotations

import warnings

_PLACEHOLDER_BACKEND_MSG = (
    "LLMEngine backend='llmir' does not run a real language model; it appends "
    "placeholder token IDs for scheduler smoke tests only. Use "
    "backend='llmir_paged' (HuggingFace + LLMIR KV cache) or backend='vllm' "
    "for real inference."
)


def warn_if_placeholder_backend(backend: str) -> None:
    if backend == "llmir":
        warnings.warn(_PLACEHOLDER_BACKEND_MSG, UserWarning, stacklevel=3)
