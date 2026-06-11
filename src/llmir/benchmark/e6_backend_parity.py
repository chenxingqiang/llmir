"""E6: multi-backend correctness parity (numpy vs torch_cuda [+ optional native])."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_factory import create_paged_kv_cache, kv_cache_backend_name


@dataclass(frozen=True)
class E6ParityConfig:
    """Decode + KV micro-benchmark settings."""

    backends: Sequence[str] = ("numpy", "torch_cuda")
    prompts: Sequence[str] = ("hello world",)
    max_new_tokens: int = 4
    seed: int = 42
    reference_backend: str = "numpy"


@dataclass
class E6BackendParityResult:
    """JSON-serializable E6 parity report."""

    experiment: str = "E6"
    mode: str = "multi_backend_correctness_parity"
    reference_backend: str = "numpy"
    backends_tested: List[str] = field(default_factory=list)
    decode_parity: List[Dict[str, Any]] = field(default_factory=list)
    kv_micro_parity: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    performance_note: str = (
        "Correctness parity only; throughput panels belong in MVP-C / E8, not E6."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _restore_kv_backend(prev: Optional[str]) -> None:
    if prev is None:
        os.environ.pop("LLMIR_KV_BACKEND", None)
    else:
        os.environ["LLMIR_KV_BACKEND"] = prev


def run_kv_micro_parity(
    kv_config: KVCacheConfig,
    *,
    backends: Sequence[str] = ("numpy", "torch_cuda"),
    seed: int = 0,
    seq_len: int = 9,
) -> Dict[str, Any]:
    """Append identical KV tensors through each backend and compare lookup output."""
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal(
        (1, seq_len, kv_config.num_heads, kv_config.head_dim), dtype=np.float32
    )
    values = rng.standard_normal(
        (1, seq_len, kv_config.num_heads, kv_config.head_dim), dtype=np.float32
    )
    seq_ids = np.array([0], dtype=np.int32)
    block_indices = np.zeros((1, kv_config.num_layers), dtype=np.int32)
    seq_lens = np.array([seq_len], dtype=np.int32)

    reference_backend = backends[0] if backends else "numpy"
    reference_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None
    rows: List[Dict[str, Any]] = []

    for backend in backends:
        prev = os.environ.get("LLMIR_KV_BACKEND")
        os.environ["LLMIR_KV_BACKEND"] = backend
        row: Dict[str, Any] = {"backend": backend}
        try:
            cache = create_paged_kv_cache(kv_config, device="cpu")
            row["kv_backend_label"] = kv_cache_backend_name(cache)
            cache.append(keys, values, seq_ids)
            k_out, v_out = cache.lookup(block_indices, seq_lens)
            k_np = _to_numpy(k_out)
            v_np = _to_numpy(v_out)
            if reference_kv is None:
                reference_kv = (k_np, v_np)
                row["matches_reference"] = True
                row["max_abs_diff_k"] = 0.0
                row["max_abs_diff_v"] = 0.0
            else:
                ref_k, ref_v = reference_kv
                row["max_abs_diff_k"] = float(np.max(np.abs(k_np - ref_k)))
                row["max_abs_diff_v"] = float(np.max(np.abs(v_np - ref_v)))
                row["matches_reference"] = (
                    row["max_abs_diff_k"] < 1e-5 and row["max_abs_diff_v"] < 1e-5
                )
        except Exception as exc:  # pragma: no cover - optional native
            row["kv_backend_label"] = "error"
            row["matches_reference"] = False
            row["error"] = str(exc)
        finally:
            _restore_kv_backend(prev)
        rows.append(row)

    all_match = all(r.get("matches_reference") for r in rows if "error" not in r)
    return {
        "reference_backend": reference_backend,
        "seq_len": seq_len,
        "backends": list(backends),
        "rows": rows,
        "all_match": all_match,
    }


def run_decode_with_backend(
    model: Any,
    tokenizer: Any,
    *,
    backend: str,
    prompt: str,
    max_new_tokens: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Greedy decode one prompt with a specific KV backend."""
    import torch

    from llmir.runtime.paged_decoder import PagedKVDecoder

    torch.manual_seed(seed)
    prev = os.environ.get("LLMIR_KV_BACKEND")
    os.environ["LLMIR_KV_BACKEND"] = backend
    try:
        decoder = PagedKVDecoder(
            model,
            tokenizer,
            enable_prefix_cache=False,
        )
        kv_label = kv_cache_backend_name(decoder._create_layer_caches()[0])
        results = decoder.decode([prompt], max_new_tokens=max_new_tokens)
        out = results[0]
        generated = list(out.generated_token_ids)
        return {
            "backend": backend,
            "kv_backend_label": kv_label,
            "prompt": prompt,
            "prompt_token_ids": list(out.prompt_token_ids),
            "generated_token_ids": generated,
            "all_token_ids": list(out.prompt_token_ids) + generated,
        }
    finally:
        _restore_kv_backend(prev)


def compare_decode_parity(
    model: Any,
    tokenizer: Any,
    cfg: Optional[E6ParityConfig] = None,
) -> List[Dict[str, Any]]:
    """Run decode on each backend and compare token ids to the reference backend."""
    cfg = cfg or E6ParityConfig()
    reference = cfg.reference_backend
    per_prompt: List[Dict[str, Any]] = []

    for prompt in cfg.prompts:
        by_backend: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        for backend in cfg.backends:
            try:
                by_backend[backend] = run_decode_with_backend(
                    model,
                    tokenizer,
                    backend=backend,
                    prompt=prompt,
                    max_new_tokens=cfg.max_new_tokens,
                    seed=cfg.seed,
                )
            except Exception as exc:  # pragma: no cover - optional native/cuda
                errors[backend] = str(exc)

        ref_tokens = (
            by_backend.get(reference, {}).get("generated_token_ids")
            if reference in by_backend
            else None
        )
        comparisons: List[Dict[str, Any]] = []
        for backend in cfg.backends:
            if backend in errors:
                comparisons.append(
                    {
                        "backend": backend,
                        "matches_reference": False,
                        "error": errors[backend],
                    }
                )
                continue
            row = by_backend[backend]
            gen = row.get("generated_token_ids", [])
            matches = ref_tokens is not None and gen == ref_tokens
            comparisons.append(
                {
                    "backend": backend,
                    "kv_backend_label": row.get("kv_backend_label"),
                    "generated_token_ids": gen,
                    "matches_reference": matches if backend != reference else True,
                }
            )

        per_prompt.append(
            {
                "prompt": prompt,
                "reference_backend": reference,
                "reference_generated_token_ids": ref_tokens,
                "comparisons": comparisons,
                "all_match": all(c.get("matches_reference") for c in comparisons),
            }
        )
    return per_prompt


def run_e6_backend_parity(
    model: Any,
    tokenizer: Any,
    *,
    kv_config: Optional[KVCacheConfig] = None,
    cfg: Optional[E6ParityConfig] = None,
) -> E6BackendParityResult:
    """Full E6: decode token parity + KV micro parity."""
    cfg = cfg or E6ParityConfig()
    kv_config = kv_config or KVCacheConfig(
        num_layers=2,
        num_heads=2,
        head_dim=8,
        block_size=4,
        max_seq_len=64,
        dtype="float32",
        enable_gpu=False,
    )

    decode_rows = compare_decode_parity(model, tokenizer, cfg)
    kv_micro = run_kv_micro_parity(kv_config, backends=cfg.backends, seed=cfg.seed)

    decode_all_match = all(row.get("all_match") for row in decode_rows)
    summary = {
        "decode_all_match": decode_all_match,
        "kv_micro_all_match": kv_micro.get("all_match", False),
        "overall_pass": decode_all_match and kv_micro.get("all_match", False),
        "backends_tested": list(cfg.backends),
    }

    return E6BackendParityResult(
        reference_backend=cfg.reference_backend,
        backends_tested=list(cfg.backends),
        decode_parity=decode_rows,
        kv_micro_parity=kv_micro,
        summary=summary,
    )


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "detach"):
        return np.asarray(data.detach().cpu().numpy())
    return np.asarray(data)
