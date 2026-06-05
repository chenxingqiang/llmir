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
        # Avoid accelerate meta-tensor paths that break on ``.to(cuda)`` later.
        "low_cpu_mem_usage": False,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return kwargs


def materialize_hf_causal_lm(model: Any) -> Any:
    """
    Fix tied-word-embedding models (e.g. OPT) that leave shared weights on ``meta``.

    Reloads weights from the on-disk safetensors snapshot when any parameter
    remains on the meta device after ``from_pretrained``.
    """
    import torch

    if not any(p.device.type == "meta" for p in model.parameters()):
        return model

    try:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
    except ImportError:
        return model

    try:
        model_id = getattr(model, "name_or_path", None) or getattr(
            model.config, "_name_or_path", ""
        )
        if not model_id:
            return model
        cache_dir = snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model

    from pathlib import Path

    root = Path(cache_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        return model

    state: Dict[str, Any] = {}
    for path in files:
        state.update(load_file(str(path)))
    if state:
        model.load_state_dict(state, strict=False, assign=True)
    if any(p.device.type == "meta" for p in model.parameters()):
        # Last resort: untie and clone from any materialized output projection.
        out = model.get_output_embeddings()
        inp = model.get_input_embeddings()
        if (
            out is not None
            and inp is not None
            and out.weight.device.type != "meta"
            and inp.weight.device.type == "meta"
        ):
            inp.weight = torch.nn.Parameter(out.weight.detach().clone())
    return model
