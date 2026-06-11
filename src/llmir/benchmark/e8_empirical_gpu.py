"""E8: optional empirical GPU bench vs HF/vLLM (B-class, not Tier-A required)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

from llmir.runtime.cuda_probe import summarize_cuda_stack


@dataclass
class E8EmpiricalConfig:
    """Small-model GPU compare defaults (lab use; not CI-required)."""

    model: str = "gpt2"
    batch_size: int = 1
    prompt_tokens: int = 32
    max_tokens: int = 8
    warmup: int = 0
    backends: tuple[str, ...] = ("hf", "llmir-paged")


@dataclass
class E8EmpiricalResult:
    """JSON-serializable E8 report."""

    experiment: str = "E8"
    mode: str = "empirical_gpu_bench"
    evidence_class: str = "B"
    status: str = "skipped"
    cuda_stack: Dict[str, Any] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    claim_scope: str = (
        "Optional same-harness throughput comparison; not a compiler correctness proof."
    )
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if not out.get("reason"):
            out.pop("reason", None)
        return out


def run_e8_empirical_gpu_bench(
    cfg: Optional[E8EmpiricalConfig] = None,
) -> E8EmpiricalResult:
    """Run GPU compare when CUDA is available; otherwise return honest skip."""
    cfg = cfg or E8EmpiricalConfig()
    stack = summarize_cuda_stack()
    result = E8EmpiricalResult(cuda_stack=stack)

    if not stack.get("torch_cuda"):
        result.status = "skipped"
        result.reason = "no_cuda_available"
        return result

    try:
        from llmir.benchmark.inference_compare import run_inference_compare
    except ImportError as exc:
        result.status = "skipped"
        result.reason = f"import_error: {exc}"
        return result

    rows: List[Dict[str, Any]] = []
    for backend in cfg.backends:
        try:
            compare_rows = run_inference_compare(
                model=cfg.model,
                batch_size=cfg.batch_size,
                prompt_tokens=cfg.prompt_tokens,
                max_tokens=cfg.max_tokens,
                warmup=cfg.warmup,
                backends=[backend],
                device="cuda",
            )
            for row in compare_rows:
                rows.append(asdict(row) if is_dataclass(row) else dict(row))
        except Exception as exc:  # pragma: no cover - lab backends
            rows.append({"backend": backend, "error": str(exc)})

    result.results = rows
    result.status = "completed" if rows else "skipped"
    if result.status == "skipped":
        result.reason = "no_backend_results"
    return result


def e8_result_to_json(result: E8EmpiricalResult) -> str:
    return json.dumps(result.to_dict(), indent=2)
