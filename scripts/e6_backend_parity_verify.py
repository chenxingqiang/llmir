#!/usr/bin/env python3
"""E6 multi-backend correctness parity — decode tokens + KV micro parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmir.benchmark.e6_backend_parity import (  # noqa: E402
    E6ParityConfig,
    run_e6_backend_parity,
)

DEFAULT_OUT = ROOT / "IEEE-conference/benchmarks/e6_backend_parity.json"


def _load_toy_model():
    """Offline 2-layer toy model (same vocab as tests/conftest tiny_llama)."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "hello": 4,
        "world": 5,
        "a": 6,
        "b": 7,
        "c": 8,
        "d": 9,
        "x": 10,
        "y": 11,
        "z": 12,
        "the": 13,
        "of": 14,
        "to": 15,
    }
    backend_tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend_tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tok,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="E6 backend correctness parity")
    parser.add_argument(
        "--backends",
        default="numpy,torch_cuda",
        help="Comma-separated LLMIR_KV_BACKEND values",
    )
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="toy",
        help="toy (offline) or HuggingFace model id (needs network)",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    cfg = E6ParityConfig(
        backends=backends,
        prompts=(args.prompt,),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    if args.model == "toy":
        model, tokenizer = _load_toy_model()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from llmir.integration.hf_load import hf_from_pretrained_kwargs, materialize_hf_causal_lm

        load_kw = hf_from_pretrained_kwargs(device="cpu", torch_dtype=torch.float32)
        model = materialize_hf_causal_lm(
            AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    result = run_e6_backend_parity(model, tokenizer, cfg=cfg)
    payload = result.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("E6 multi-backend correctness parity")
    print("=" * 50)
    print(f"backends: {', '.join(backends)}")
    print(f"decode_all_match: {payload['summary']['decode_all_match']}")
    print(f"kv_micro_all_match: {payload['summary']['kv_micro_all_match']}")
    print(f"overall_pass: {payload['summary']['overall_pass']}")
    print(f"Wrote {args.output}")
    return 0 if payload["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
