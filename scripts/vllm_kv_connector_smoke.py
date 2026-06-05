#!/usr/bin/env python3
"""Validate LLMIRKVStorage and optional vLLM connector registration."""

from __future__ import annotations

import argparse
import tempfile

import numpy as np

from llmir.integration.vllm_connector import (
    CONNECTOR_NAME,
    is_vllm_connector_available,
    register_llmir_vllm_connector,
)
from llmir.integration.vllm_kv_storage import LLMIRKVStorage, LLMIRKVStorageConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        storage = LLMIRKVStorage(LLMIRKVStorageConfig(storage_path=tmp))
        tokens = [1, 2, 3, 4, 5]
        kv = np.random.randn(5, 2, 8).astype(np.float32)
        storage.store_layer_kv(tokens, "layer0", kv)
        loaded = storage.load_layer_kv(tokens, "layer0")
        assert np.allclose(kv, loaded), "roundtrip failed"
        assert storage.longest_cached_prefix_length(tokens[:4]) == 4
        print("LLMIRKVStorage roundtrip OK")

    print(f"vLLM connector API available: {is_vllm_connector_available()}")
    if args.register:
        ok = register_llmir_vllm_connector()
        print(f"register {CONNECTOR_NAME}: {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
