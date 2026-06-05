"""Prefix-cache integration tests for PagedKVDecoder."""

import pytest


def test_warm_prefix_skips_prefill_on_shared_prompt(tiny_llama):
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    from llmir.runtime.paged_decoder import PagedKVDecoder

    decoder = PagedKVDecoder(model, tokenizer)
    # Four tokens — matches decoder default ``min_prefix_length=4``.
    system = "hello hello hello hello"
    user_a = " a"
    user_b = " b"

    warmed = decoder.warm_prefix(system)
    assert warmed == 4

    [first] = decoder.decode([system + user_a], max_new_tokens=2)
    assert first.prefix_hit_tokens >= warmed
    assert first.prefill_tokens_computed == 1

    [second] = decoder.decode([system + user_b], max_new_tokens=2)
    assert second.prefix_hit_tokens >= warmed
    assert second.prefill_tokens_computed == 1

    stats = decoder.prefix_cache_stats
    assert stats is not None
    assert stats.hits >= 2


def test_decode_populates_prefix_cache_without_warm(tiny_llama):
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    from llmir.runtime.paged_decoder import PagedKVDecoder

    decoder = PagedKVDecoder(model, tokenizer)
    system = "hello hello hello hello"

    [cold] = decoder.decode([system + " a"], max_new_tokens=1)
    assert cold.prefix_hit_tokens == 0
    assert cold.prefill_tokens_computed == 5

    [warm] = decoder.decode([system + " b"], max_new_tokens=1)
    assert warm.prefix_hit_tokens >= 4
    assert warm.prefill_tokens_computed < cold.prefill_tokens_computed


def test_prefix_cache_preserves_greedy_tokens(tiny_llama):
    pytest.importorskip("torch")
    model, tokenizer, _ = tiny_llama
    from llmir.runtime.paged_decoder import PagedKVDecoder

    decoder_plain = PagedKVDecoder(
        model, tokenizer, enable_prefix_cache=False
    )
    decoder_cached = PagedKVDecoder(model, tokenizer)

    prompt = "hello hello hello hello a"
    plain = decoder_plain.decode([prompt], max_new_tokens=2)[0]
    decoder_cached.warm_prefix("hello hello hello hello")
    cached = decoder_cached.decode([prompt], max_new_tokens=2)[0]

    assert plain.generated_token_ids == cached.generated_token_ids
