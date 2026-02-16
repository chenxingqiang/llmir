"""
LLMIR Integration Module

Adapters for external frameworks (HuggingFace Transformers, etc.).
"""

__all__ = []

try:
    from llmir.integration.huggingface import from_pretrained
    __all__.append("from_pretrained")
except ImportError:
    # transformers not installed
    pass
