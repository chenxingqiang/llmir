"""Tests for placeholder backend warnings."""

import warnings

from llmir import LLMEngine


def test_llmir_backend_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMEngine(model_path="test-model", backend="llmir")
    assert any("placeholder" in str(w.message).lower() for w in caught)
