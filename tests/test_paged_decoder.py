"""Tests for the LLMIR_PAGED kernel-integrated decode path."""

from __future__ import annotations

import sys
from typing import Any, List

import pytest

from llmir.runtime.paged_decoder import kv_config_from_hf_config
from llmir.serving.config import BackendType, SamplingParams
from llmir.serving.engine import LLMEngine


class _FakeHFConfig:
    """Plain object that mimics relevant HF config attributes."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_kv_config_from_hf_config_uses_hidden_size_when_no_head_dim() -> None:
    """head_dim falls back to ``hidden_size // num_attention_heads`` when absent."""

    cfg = _FakeHFConfig(
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=512,
        max_position_embeddings=1024,
    )

    kv = kv_config_from_hf_config(cfg, dtype="float32")

    assert kv.num_layers == 4
    assert kv.num_heads == 4  # KV heads (GQA) preferred over attention heads
    assert kv.head_dim == 64  # 512 / 8
    assert kv.max_seq_len == 1024
    assert kv.dtype == "float32"
    assert kv.enable_gpu is False


def test_kv_config_from_hf_config_falls_back_when_attrs_missing() -> None:
    """An empty config still yields a usable KVCacheConfig with safe defaults."""

    kv = kv_config_from_hf_config(_FakeHFConfig())

    assert kv.num_layers >= 1
    assert kv.num_heads >= 1
    assert kv.head_dim >= 1
    assert kv.max_seq_len >= 1


def test_llm_engine_routes_llmir_paged_to_decoder(monkeypatch) -> None:
    """``LLMEngine.generate`` dispatches to ``_generate_llmir_paged`` for the new backend."""

    from llmir.runtime.paged_decoder import DecodeResult

    class _StubDecoder:
        last_call: dict = {}

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise AssertionError("Real decoder should not be constructed in this test")

        def decode(
            self,
            prompts: List[str],
            *,
            max_new_tokens: int,
            stop_token_ids=None,
        ) -> List[DecodeResult]:
            _StubDecoder.last_call = {
                "prompts": list(prompts),
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": list(stop_token_ids or ()),
            }
            return [
                DecodeResult(
                    prompt_token_ids=[1, 2, 3],
                    generated_token_ids=[10, 11, 12][:max_new_tokens],
                    text="hi",
                    finish_reason="length",
                )
                for _ in prompts
            ]

    engine = LLMEngine(model_path="test-model", backend=BackendType.LLMIR_PAGED)

    # Bypass the real model load: install a pre-built stub decoder so
    # ``_ensure_llmir_paged`` is a no-op for the rest of the test.
    engine._paged_decoder = _StubDecoder.__new__(_StubDecoder)
    engine._paged_decoder.decode = _StubDecoder.decode.__get__(engine._paged_decoder)

    outputs = engine.generate(
        ["hello", "world"],
        SamplingParams(max_tokens=2, stop_token_ids=[42]),
    )

    assert _StubDecoder.last_call["prompts"] == ["hello", "world"]
    assert _StubDecoder.last_call["max_new_tokens"] == 2
    assert _StubDecoder.last_call["stop_token_ids"] == [42]
    assert len(outputs) == 2
    for output in outputs:
        assert output.finished
        assert output.outputs[0].token_ids == [10, 11]
        assert output.outputs[0].text == "hi"
        assert output.outputs[0].finish_reason == "length"


def test_llm_engine_llmir_paged_requires_transformers(monkeypatch) -> None:
    """Without transformers installed, the kernel-integrated path raises a clear error."""

    # Simulate ``transformers`` being unavailable for both top-level and
    # submodule imports used by ``_ensure_llmir_paged``.
    monkeypatch.setitem(sys.modules, "transformers", None)

    engine = LLMEngine(model_path="test-model", backend=BackendType.LLMIR_PAGED)

    with pytest.raises(ImportError, match="transformers"):
        engine.generate("hello", SamplingParams(max_tokens=1))


def test_normalize_backend_accepts_llmir_paged() -> None:
    """The new backend value is recognized via both string and enum forms."""

    engine_a = LLMEngine(model_path="test-model", backend="llmir_paged")
    engine_b = LLMEngine(model_path="test-model", backend=BackendType.LLMIR_PAGED)

    assert engine_a.backend == "llmir_paged"
    assert engine_b.backend == "llmir_paged"


# ---------------------------------------------------------------------------
# Real-model integration tests (gated on torch + transformers being installed)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_llama():
    """Build a 2-layer toy LLaMA model + a synthetic tokenizer in-memory.

    The model's weights are random and the tokenizer is a tiny word-level
    one built locally so the test runs fully offline. We only assert
    structural things (correct number of tokens generated, K/V routed
    through PagedKVCache) — never specific token values.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    # 16-entry vocab covers the test prompts ("hello", "a") plus controls.
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


def test_paged_decoder_routes_kv_through_paged_kv_cache(tiny_llama, monkeypatch):
    """Real decode loop puts every layer's K/V into ``PagedKVCache``.

    This is the load-bearing test for "actually intervening at the kernel
    layer": we spy on ``PagedKVCache.append`` and assert it is called once
    per layer per step.
    """

    pytest.importorskip("torch")
    model, tokenizer, config = tiny_llama

    from llmir.runtime import kv_cache as kv_cache_module
    from llmir.runtime.paged_decoder import PagedKVDecoder

    call_log: list = []
    real_append = kv_cache_module.PagedKVCache.append

    def spy_append(self, keys, values, seq_ids):
        call_log.append((id(self), keys.shape, values.shape))
        return real_append(self, keys, values, seq_ids)

    monkeypatch.setattr(kv_cache_module.PagedKVCache, "append", spy_append)

    decoder = PagedKVDecoder(model, tokenizer)

    # Sanity check: kv_config matches the model.
    assert decoder.kv_config.num_layers == config.num_hidden_layers
    assert decoder.kv_config.num_heads == config.num_key_value_heads

    max_new = 3
    results = decoder.decode(["hello"], max_new_tokens=max_new)

    assert len(results) == 1
    out = results[0]
    # Generated some tokens (may stop early on EOS, but with random weights and
    # vocab=128 + max_new=3 it's overwhelmingly unlikely).
    assert 1 <= len(out.generated_token_ids) <= max_new
    assert isinstance(out.text, str)

    # ``append`` must have been called: prefill (1x num_layers) + each decode
    # step that ran (n_steps x num_layers). At minimum: num_layers calls.
    layer_count = config.num_hidden_layers
    assert len(call_log) >= layer_count
    assert len(call_log) % layer_count == 0  # whole steps only
    n_steps = len(call_log) // layer_count
    # We did prefill + at most (max_new - 1) decode steps; n_steps >= 1.
    assert 1 <= n_steps <= max_new


def test_paged_decoder_decode_result_round_trips_tokens(tiny_llama):
    """End-to-end: tokens decoded match the result's token_ids."""

    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    from llmir.runtime.paged_decoder import PagedKVDecoder

    decoder = PagedKVDecoder(model, tokenizer)
    [result] = decoder.decode(["a"], max_new_tokens=2)
    assert result.text == tokenizer.decode(
        result.generated_token_ids, skip_special_tokens=True
    )
