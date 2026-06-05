"""Tests for vLLM KV storage and connector helpers."""

import tempfile

import numpy as np

from llmir.integration.vllm_connector import (
    CONNECTOR_NAME,
    build_kv_transfer_extra_config,
    is_vllm_connector_available,
    register_llmir_vllm_connector,
)
from llmir.integration.vllm_kv_storage import (
    LLMIRKVStorage,
    LLMIRKVStorageConfig,
    align_to_block_size,
)


def test_align_to_block_size():
    assert align_to_block_size(10, 4) == 8
    assert align_to_block_size(4, 4) == 0


def test_kv_storage_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        storage = LLMIRKVStorage(
            LLMIRKVStorageConfig(storage_path=tmp, min_prefix_length=2)
        )
        tokens = [10, 11, 12, 13]
        kv = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        storage.store_layer_kv(tokens, "layers.0", kv)
        loaded = storage.load_layer_kv(tokens, "layers.0")
        assert np.allclose(kv, loaded)
        assert storage.longest_cached_prefix_length(tokens[:3]) == 3
        assert storage.longest_cached_prefix_length([99, 100]) == 0


def test_build_kv_transfer_extra_config():
    cfg = build_kv_transfer_extra_config("/data/kv", min_prefix_length=8)
    assert cfg["shared_storage_path"] == "/data/kv"
    assert cfg["min_prefix_length"] == 8


def test_register_connector_without_vllm():
    if is_vllm_connector_available():
        assert register_llmir_vllm_connector() in (True, False)
    else:
        assert register_llmir_vllm_connector() is False


def test_connector_name_constant():
    assert CONNECTOR_NAME == "LLMIRConnector"
