"""
LLMIR Model Importers.

This package provides importers for converting models from various frameworks
(PyTorch, ONNX, HuggingFace) into the LLMIR dialect.
"""

from llmir.importers.pytorch import (
    AttentionPattern,
    ImportConfig,
    ImportMode,
    LinearPattern,
    PatternMatcher,
    PyTorchImporter,
    import_pytorch_model,
)
from llmir.importers.toy_attention import import_toy_attention_to_mlir

__all__ = [
    "ImportConfig",
    "ImportMode",
    "PyTorchImporter",
    "import_pytorch_model",
    "import_toy_attention_to_mlir",
    "PatternMatcher",
    "AttentionPattern",
    "LinearPattern",
]
