"""Standard decoder workload buckets S1/S2/S3 (see docs/DECODER_WORKLOAD_ARCHITECTURES.md)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmir.benchmark.sharegpt_prefix_bench import (
    ShareGPTPrefixBenchConfig,
    run_sharegpt_prefix_benchmark,
)


@dataclass(frozen=True)
class DecoderWorkloadBucket:
    """One S1/S2/S3 shape with a committed sim JSON artifact name."""

    bucket_id: str
    label: str
    system_prompt_tokens: int
    user_suffix_tokens: int
    num_requests: int
    artifact_name: str

    def to_config(self) -> ShareGPTPrefixBenchConfig:
        return ShareGPTPrefixBenchConfig(
            system_prompt_tokens=self.system_prompt_tokens,
            num_requests=self.num_requests,
            user_suffix_tokens=self.user_suffix_tokens,
            max_new_tokens=4,
            model="gpt2",
            device="auto",
            warmup_requests=0,
        )

    def artifact_path(self, benchmarks_dir: Path) -> Path:
        return benchmarks_dir / self.artifact_name


DECODER_WORKLOAD_BUCKETS: Dict[str, DecoderWorkloadBucket] = {
    "S1": DecoderWorkloadBucket(
        bucket_id="S1",
        label="short multi-tenant",
        system_prompt_tokens=128,
        user_suffix_tokens=16,
        num_requests=32,
        artifact_name="shared_prefix_decoder_128_sim.json",
    ),
    "S2": DecoderWorkloadBucket(
        bucket_id="S2",
        label="RAG shared system",
        system_prompt_tokens=2048,
        user_suffix_tokens=8,
        num_requests=32,
        artifact_name="shared_prefix_decoder_2048_sim.json",
    ),
    "S3": DecoderWorkloadBucket(
        bucket_id="S3",
        label="long document prefill",
        system_prompt_tokens=8192,
        user_suffix_tokens=64,
        num_requests=16,
        artifact_name="shared_prefix_decoder_8192_sim.json",
    ),
}


def list_decoder_workload_buckets() -> List[DecoderWorkloadBucket]:
    return list(DECODER_WORKLOAD_BUCKETS.values())


def generate_bucket_sim_payload(bucket: DecoderWorkloadBucket) -> Dict[str, Any]:
    """CPU KV simulation only (no HuggingFace)."""
    payload = run_sharegpt_prefix_benchmark(
        bucket.to_config(),
        run_simulation=True,
        run_llmir_paged=False,
    )
    payload["bucket_id"] = bucket.bucket_id
    payload["bucket_label"] = bucket.label
    return payload


def write_bucket_artifact(
    bucket: DecoderWorkloadBucket,
    benchmarks_dir: Path,
) -> Path:
    import json

    out = bucket.artifact_path(benchmarks_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = generate_bucket_sim_payload(bucket)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def verify_bucket_artifact(
    bucket: DecoderWorkloadBucket,
    benchmarks_dir: Path,
) -> Optional[str]:
    """Return error message when missing or config mismatch."""
    path = bucket.artifact_path(benchmarks_dir)
    if not path.is_file():
        return f"missing {path.name}"

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    expected = asdict(bucket.to_config())
    for key in (
        "system_prompt_tokens",
        "num_requests",
        "user_suffix_tokens",
        "max_new_tokens",
        "model",
    ):
        if cfg.get(key) != expected[key]:
            return f"{path.name}: config.{key}={cfg.get(key)!r} expected {expected[key]!r}"
    if data.get("mode") != "shared_prefix_decoder":
        return f"{path.name}: mode must be shared_prefix_decoder"
    if len(data.get("results", [])) < 2:
        return f"{path.name}: expected >=2 simulation rows"
    return None
