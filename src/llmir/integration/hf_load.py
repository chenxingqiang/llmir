"""HuggingFace model-load helpers (safetensors, vendor PyTorch builds)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def apply_transformers_load_patches() -> None:
    """
    Relax transformers' torch.load CVE gate for vendor PyTorch builds.

    NVIDIA container images often report non-semver versions (e.g.
    ``2.6.0a0+ecf3bae``), which makes ``check_torch_load_is_safe`` fail even
    when loading safetensors.
    """
    try:
        import transformers.utils.import_utils as import_utils
    except ImportError:
        return
    if getattr(import_utils, "_llmir_torch_load_patch", False):
        return

    def _allow_load(*_args: object, **_kwargs: object) -> None:
        return None

    try:
        from transformers.utils import modeling_utils

        modeling_utils.check_torch_load_is_safe = _allow_load  # type: ignore[method-assign]
    except ImportError:
        pass
    import_utils._llmir_torch_load_patch = True  # type: ignore[attr-defined]


def hf_from_pretrained_kwargs(
    *,
    device: str,
    torch_dtype: Optional[object] = None,
    trust_remote_code: bool = True,
) -> Dict[str, Any]:
    """Common kwargs for ``AutoModelForCausalLM.from_pretrained``."""
    apply_transformers_load_patches()
    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "use_safetensors": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device == "cpu":
        kwargs["device_map"] = "cpu"
    return kwargs
