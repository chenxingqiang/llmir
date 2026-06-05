"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def tiny_llama():
    """2-layer toy LLaMA + local tokenizer (offline)."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
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
    backend_tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend_tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend_tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )

    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    return model, tokenizer, config
