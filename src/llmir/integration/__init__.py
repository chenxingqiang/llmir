"""
LLMIR Integration Module

Adapters for external frameworks (HuggingFace Transformers, vLLM, etc.).
"""

__all__ = [
    "build_kv_transfer_extra_config",
    "is_vllm_connector_available",
    "register_llmir_vllm_connector",
    "LLMIRKVStorage",
    "LLMIRKVStorageConfig",
]

from llmir.integration.vllm_connector import (
    build_kv_transfer_extra_config,
    is_vllm_connector_available,
    register_llmir_vllm_connector,
)
from llmir.integration.vllm_kv_storage import LLMIRKVStorage, LLMIRKVStorageConfig

try:
    from llmir.integration.huggingface import from_pretrained  # noqa: F401

    __all__.append("from_pretrained")
except ImportError:
    pass
