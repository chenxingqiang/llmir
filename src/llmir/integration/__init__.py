"""
LLMIR Integration Module

Adapters for external frameworks (HuggingFace Transformers, etc.).
"""

try:
    from llmir.integration.huggingface import from_pretrained
except ImportError:
    # transformers not installed
    __all__ = []
else:
    __all__ = ["from_pretrained"]
