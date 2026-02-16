"""Integration tests for HuggingFace Transformers."""

import pytest

pytest.importorskip("transformers", reason="Requires transformers (pip install llmir[full])")


@pytest.mark.network
class TestFromPretrained:
    """Tests for from_pretrained with real HuggingFace configs.

    Requires network access to download configs from HuggingFace Hub.
    Use public models to avoid authentication (gpt2, cosmo-1b).
    """

    def test_from_pretrained_gpt2(self):
        """Test loading config from GPT-2 (small, public model)."""
        from llmir import from_pretrained

        optimizer = from_pretrained("gpt2")
        kv_config = optimizer.get_optimized_kv_cache_config()

        assert optimizer.config.num_layers == 12
        assert optimizer.config.hidden_size == 768
        assert kv_config.num_layers == 12
        assert kv_config.num_heads == 12
        assert kv_config.head_dim == 64

    def test_from_pretrained_llama_arch(self):
        """Test loading config from Llama-architecture model (Cosmo, public)."""
        from llmir import from_pretrained

        optimizer = from_pretrained("HuggingFaceTB/cosmo-1b")
        kv_config = optimizer.get_optimized_kv_cache_config()

        assert optimizer.config.num_layers == 24
        assert optimizer.config.hidden_size == 2048
        assert kv_config.num_layers == 24
        assert kv_config.head_dim == 128

    def test_from_pretrained_returns_optimizer(self):
        """Test that from_pretrained returns usable ModelOptimizer."""
        from llmir import from_pretrained

        optimizer = from_pretrained("gpt2")
        quant_config = optimizer.get_recommended_quant_config()
        mem = optimizer.estimate_memory(batch_size=1, seq_len=128)

        assert quant_config is not None
        assert mem > 0
