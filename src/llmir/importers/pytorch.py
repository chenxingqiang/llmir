"""
PyTorch Model Importer for LLMIR.

This module provides infrastructure for importing PyTorch models into the
LLMIR dialect, enabling compiler-level optimization of LLM inference workloads.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ImportMode(Enum):
    """Import mode for PyTorch models."""
    EAGER = auto()      # Trace eager mode model
    SCRIPT = auto()     # Use TorchScript
    EXPORT = auto()     # Use torch.export (PyTorch 2.0+)
    FX = auto()         # Use FX graph capture


@dataclass
class ImportConfig:
    """Configuration for PyTorch model import.
    
    Attributes:
        mode: Import mode to use (EAGER, SCRIPT, EXPORT, FX)
        target_dialect: Target MLIR dialect ("llm", "linalg", "tosa")
        enable_kv_cache: Whether to insert KV cache operations
        enable_quantization: Whether to detect and preserve quantization
        enable_parallelism: Whether to detect parallelism opportunities
        batch_size: Sample batch size for tracing (None for dynamic)
        seq_length: Sample sequence length for tracing (None for dynamic)
        max_seq_length: Maximum sequence length for KV cache
        block_size: KV cache block size (None for auto-optimization)
        dtype: Target data type (fp16, bf16, fp32)
        device: Target device for compilation (cuda, cpu)
    """
    mode: ImportMode = ImportMode.EAGER
    target_dialect: str = "llm"
    enable_kv_cache: bool = True
    enable_quantization: bool = True
    enable_parallelism: bool = False
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    max_seq_length: int = 8192
    block_size: Optional[int] = None
    dtype: str = "fp16"
    device: str = "cuda"


@dataclass 
class AttentionPattern:
    """Detected attention pattern in PyTorch model.
    
    Used for pattern matching during import to convert PyTorch
    attention implementations to llm.attention or llm.paged_attention.
    """
    pattern_type: str  # "sdpa", "flash", "custom", "multi_head"
    num_heads: int
    head_dim: int
    has_mask: bool
    is_causal: bool
    query_node: Any = None
    key_node: Any = None
    value_node: Any = None
    output_node: Any = None


@dataclass
class LinearPattern:
    """Detected linear/matmul pattern in PyTorch model.
    
    Used for detecting quantizable operations and weight matrices
    that can be represented with llm.quantized_matmul.
    """
    weight_shape: Tuple[int, ...]
    has_bias: bool
    is_quantized: bool
    quantization_bits: Optional[int] = None
    quantization_group_size: Optional[int] = None
    input_node: Any = None
    weight_node: Any = None
    output_node: Any = None


class PatternMatcher:
    """Pattern matcher for identifying LLM-specific patterns in PyTorch graphs."""
    
    def __init__(self):
        self._attention_patterns: List[AttentionPattern] = []
        self._linear_patterns: List[LinearPattern] = []
        
    def find_attention_patterns(self, graph) -> List[AttentionPattern]:
        """Find attention patterns in a traced graph.
        
        Detects:
        - torch.nn.functional.scaled_dot_product_attention
        - torch.nn.MultiheadAttention
        - Custom QKV projection + matmul + softmax patterns
        - FlashAttention patterns
        
        Args:
            graph: PyTorch FX graph or TorchScript graph
            
        Returns:
            List of detected attention patterns
        """
        patterns = []
        
        # Pattern 1: F.scaled_dot_product_attention
        for node in self._iter_nodes(graph):
            if self._is_sdpa_call(node):
                pattern = self._extract_sdpa_pattern(node)
                if pattern:
                    patterns.append(pattern)
                    
        # Pattern 2: Manual attention (Q @ K.T / sqrt(d) -> softmax -> @ V)
        for node in self._iter_nodes(graph):
            if self._is_softmax(node):
                pattern = self._extract_manual_attention_pattern(node, graph)
                if pattern:
                    patterns.append(pattern)
                    
        # Pattern 3: nn.MultiheadAttention
        for node in self._iter_nodes(graph):
            if self._is_mha_module(node):
                pattern = self._extract_mha_pattern(node)
                if pattern:
                    patterns.append(pattern)
        
        self._attention_patterns = patterns
        return patterns
    
    def find_linear_patterns(self, graph) -> List[LinearPattern]:
        """Find linear/matmul patterns in a traced graph.
        
        Detects:
        - torch.nn.Linear modules
        - torch.matmul operations
        - Quantized linear layers (GPTQ, AWQ, etc.)
        
        Args:
            graph: PyTorch FX graph or TorchScript graph
            
        Returns:
            List of detected linear patterns
        """
        patterns = []
        
        for node in self._iter_nodes(graph):
            if self._is_linear(node):
                pattern = self._extract_linear_pattern(node)
                if pattern:
                    patterns.append(pattern)
            elif self._is_quantized_linear(node):
                pattern = self._extract_quantized_linear_pattern(node)
                if pattern:
                    patterns.append(pattern)
        
        self._linear_patterns = patterns
        return patterns
    
    def _iter_nodes(self, graph):
        """Iterate over nodes in a graph (handles FX and TorchScript)."""
        if hasattr(graph, 'nodes'):
            # FX graph
            return graph.nodes
        elif hasattr(graph, 'graph'):
            # Module with graph
            return graph.graph.nodes
        return []
    
    def _is_sdpa_call(self, node) -> bool:
        """Check if node is a scaled_dot_product_attention call."""
        if hasattr(node, 'target'):
            target = str(node.target)
            return 'scaled_dot_product_attention' in target
        return False
    
    def _is_softmax(self, node) -> bool:
        """Check if node is a softmax operation."""
        if hasattr(node, 'target'):
            target = str(node.target)
            return 'softmax' in target.lower()
        return False
    
    def _is_mha_module(self, node) -> bool:
        """Check if node is a MultiheadAttention module call."""
        if hasattr(node, 'target'):
            target = str(node.target)
            return 'multihead' in target.lower() and 'attention' in target.lower()
        return False
    
    def _is_linear(self, node) -> bool:
        """Check if node is a linear/matmul operation."""
        if hasattr(node, 'target'):
            target = str(node.target)
            return 'linear' in target.lower() or 'matmul' in target.lower()
        return False
    
    def _is_quantized_linear(self, node) -> bool:
        """Check if node is a quantized linear operation (GPTQ, AWQ, etc.)."""
        if hasattr(node, 'target'):
            target = str(node.target)
            return any(q in target.lower() for q in ['quantized', 'gptq', 'awq', 'bnb'])
        return False
    
    def _extract_sdpa_pattern(self, node) -> Optional[AttentionPattern]:
        """Extract attention pattern from SDPA node."""
        try:
            args = node.args
            if len(args) >= 3:
                return AttentionPattern(
                    pattern_type="sdpa",
                    num_heads=-1,  # Will be inferred from shape
                    head_dim=-1,
                    has_mask=len(args) > 3,
                    is_causal=getattr(node, 'is_causal', False),
                    query_node=args[0],
                    key_node=args[1],
                    value_node=args[2],
                    output_node=node
                )
        except Exception as e:
            logger.warning(f"Failed to extract SDPA pattern: {e}")
        return None
    
    def _extract_manual_attention_pattern(self, softmax_node, graph) -> Optional[AttentionPattern]:
        """Extract manual attention pattern from softmax node."""
        # This is a simplified pattern matcher
        # A full implementation would trace back through the graph
        return None
    
    def _extract_mha_pattern(self, node) -> Optional[AttentionPattern]:
        """Extract attention pattern from MultiheadAttention module."""
        return None
    
    def _extract_linear_pattern(self, node) -> Optional[LinearPattern]:
        """Extract linear pattern from node."""
        try:
            return LinearPattern(
                weight_shape=(-1, -1),  # Will be inferred
                has_bias=True,
                is_quantized=False,
                input_node=node.args[0] if node.args else None,
                output_node=node
            )
        except Exception:
            return None
    
    def _extract_quantized_linear_pattern(self, node) -> Optional[LinearPattern]:
        """Extract quantized linear pattern from node."""
        try:
            return LinearPattern(
                weight_shape=(-1, -1),
                has_bias=True,
                is_quantized=True,
                quantization_bits=4,  # Default, will be detected
                input_node=node.args[0] if node.args else None,
                output_node=node
            )
        except Exception:
            return None


class MLIRBuilder:
    """Builder for constructing MLIR from patterns.
    
    Generates MLIR text representation of LLM dialect operations
    from detected PyTorch patterns.
    """
    
    def __init__(self, config: ImportConfig):
        self.config = config
        self._operations: List[str] = []
        self._types: Dict[str, str] = {}
        self._next_id = 0
        
    def build_module(self, 
                     attention_patterns: List[AttentionPattern],
                     linear_patterns: List[LinearPattern],
                     model_config: Optional[Dict[str, Any]] = None) -> str:
        """Build MLIR module from detected patterns.
        
        Args:
            attention_patterns: Detected attention patterns
            linear_patterns: Detected linear patterns  
            model_config: Model configuration (num_layers, hidden_size, etc.)
            
        Returns:
            MLIR module as string
        """
        lines = []
        lines.append("module {")
        
        # Add function declaration
        lines.append(self._build_function_signature(model_config))
        
        # Build operations
        for pattern in attention_patterns:
            lines.append(self._build_attention_op(pattern))
            
        for pattern in linear_patterns:
            if pattern.is_quantized:
                lines.append(self._build_quantized_matmul(pattern))
            else:
                lines.append(self._build_linear_op(pattern))
        
        lines.append("  }")  # End function
        lines.append("}")  # End module
        
        return "\n".join(lines)
    
    def _build_function_signature(self, model_config: Optional[Dict[str, Any]]) -> str:
        """Build function signature with KV cache if enabled."""
        batch = self.config.batch_size or "?"
        seq = self.config.seq_length or "?"
        hidden = model_config.get("hidden_size", 4096) if model_config else 4096
        dtype = "f16" if self.config.dtype == "fp16" else "f32"
        
        args = [f"    %input: tensor<{batch}x{seq}x{hidden}x{dtype}>"]
        
        if self.config.enable_kv_cache:
            num_layers = model_config.get("num_layers", 32) if model_config else 32
            num_heads = model_config.get("num_attention_heads", 32) if model_config else 32
            head_dim = model_config.get("head_dim", 128) if model_config else 128
            block_size = self.config.block_size or 16
            max_seq = self.config.max_seq_length
            
            args.append(f"    %kv_cache: !llm.paged_kv_cache<{dtype}, {num_layers}, {num_heads}, {head_dim}, {block_size}, {max_seq}>")
            args.append(f"    %block_indices: tensor<{batch}x128xi32>")
            args.append(f"    %seq_lens: tensor<{batch}xi32>")
        
        result_type = f"tensor<{batch}x{seq}x{hidden}x{dtype}>"
        
        return f"""  func.func @forward(
{chr(10).join(args)}
  ) -> {result_type} {{"""
    
    def _build_attention_op(self, pattern: AttentionPattern) -> str:
        """Build llm.attention or llm.paged_attention operation."""
        vid = self._next_var_id()
        
        if self.config.enable_kv_cache:
            # Use paged attention
            return f"""    %attn_{vid} = llm.paged_attention %query_{vid}, %kv_cache, %block_indices, %seq_lens {{
      num_heads = {pattern.num_heads} : i32,
      head_dim = {pattern.head_dim} : i32,
      scale = {1.0 / (pattern.head_dim ** 0.5) if pattern.head_dim > 0 else 0.125:.6f} : f32
    }} : (tensor<?x?x{pattern.num_heads}x{pattern.head_dim if pattern.head_dim > 0 else 128}xf16>, !llm.paged_kv_cache<...>, tensor<?x?xi32>, tensor<?xi32>) -> tensor<?x?x{pattern.num_heads}x{pattern.head_dim if pattern.head_dim > 0 else 128}xf16>"""
        else:
            # Use standard attention
            causal_str = "true" if pattern.is_causal else "false"
            return f"""    %attn_{vid} = llm.attention %query_{vid}, %key_{vid}, %value_{vid} {{
      scale = {1.0 / (pattern.head_dim ** 0.5) if pattern.head_dim > 0 else 0.125:.6f} : f32,
      causal = {causal_str}
    }} : tensor<?x?x{pattern.num_heads}x{pattern.head_dim if pattern.head_dim > 0 else 128}xf16>, tensor<?x?x{pattern.num_heads}x{pattern.head_dim if pattern.head_dim > 0 else 128}xf16>, tensor<?x?x{pattern.num_heads}x{pattern.head_dim if pattern.head_dim > 0 else 128}xf16> -> tensor<?x?x{pattern.num_heads}x{pattern.head_dim if pattern.head_dim > 0 else 128}xf16>"""
    
    def _build_linear_op(self, pattern: LinearPattern) -> str:
        """Build linalg.matmul or custom linear operation."""
        vid = self._next_var_id()
        return f"""    %linear_{vid} = linalg.matmul ins(%input_{vid}, %weight_{vid} : tensor<?x?xf16>, tensor<?x?xf16>) outs(%output_{vid} : tensor<?x?xf16>) -> tensor<?x?xf16>"""
    
    def _build_quantized_matmul(self, pattern: LinearPattern) -> str:
        """Build llm.quantized_matmul operation."""
        vid = self._next_var_id()
        bits = pattern.quantization_bits or 8
        group_size = pattern.quantization_group_size or 128
        return f"""    %qmatmul_{vid} = llm.quantized_matmul %input_{vid}, %q_weight_{vid}, %scales_{vid} {{
      bits = {bits} : i32,
      group_size = {group_size} : i64
    }} : tensor<?x?xf16>, !llm.quantized_tensor<i{bits}, [?, ?], true, false, -1, {group_size}, {bits}>, tensor<?xf32> -> tensor<?x?xf16>"""
    
    def _next_var_id(self) -> int:
        """Get next variable ID."""
        self._next_id += 1
        return self._next_id


class PyTorchImporter:
    """Main importer class for PyTorch models.
    
    Imports PyTorch models into LLMIR dialect, detecting attention patterns,
    linear operations, and other LLM-specific patterns.
    
    Example:
        >>> from llmir.importers import PyTorchImporter
        >>> from transformers import AutoModelForCausalLM
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> importer = PyTorchImporter()
        >>> mlir_module = importer.import_model(model)
    """
    
    def __init__(self, config: Optional[ImportConfig] = None):
        """Initialize the importer.
        
        Args:
            config: Import configuration. Uses defaults if not provided.
        """
        self.config = config or ImportConfig()
        self.pattern_matcher = PatternMatcher()
        self._traced_graph = None
        self._model_config: Dict[str, Any] = {}
        
    def import_model(
        self,
        model,
        sample_inputs: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Import a PyTorch model to MLIR.
        
        Args:
            model: PyTorch model (nn.Module or traced)
            sample_inputs: Sample inputs for tracing (required for eager mode)
            model_config: Optional model configuration override
            
        Returns:
            MLIR module as string
            
        Raises:
            ImportError: If required dependencies are missing
            ValueError: If tracing fails or model cannot be imported
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch import requires torch. Install with: pip install torch"
            )
        
        # Store model config
        if model_config:
            self._model_config = model_config
        else:
            self._model_config = self._extract_model_config(model)
        
        # Trace the model
        graph = self._trace_model(model, sample_inputs)
        
        # Find patterns
        attention_patterns = self.pattern_matcher.find_attention_patterns(graph)
        linear_patterns = self.pattern_matcher.find_linear_patterns(graph)
        
        logger.info(f"Found {len(attention_patterns)} attention patterns")
        logger.info(f"Found {len(linear_patterns)} linear patterns")
        
        # Build MLIR
        builder = MLIRBuilder(self.config)
        mlir_text = builder.build_module(
            attention_patterns, 
            linear_patterns,
            self._model_config
        )
        
        return mlir_text
    
    def _trace_model(self, model, sample_inputs: Optional[Dict[str, Any]]):
        """Trace the model to get a graph representation.
        
        Args:
            model: PyTorch model
            sample_inputs: Sample inputs for tracing
            
        Returns:
            Traced graph (FX GraphModule or TorchScript)
        """
        import torch
        
        if self.config.mode == ImportMode.FX:
            from torch.fx import symbolic_trace
            try:
                graph = symbolic_trace(model)
                self._traced_graph = graph
                return graph
            except Exception as e:
                logger.warning(f"FX tracing failed, falling back to eager: {e}")
                
        if self.config.mode == ImportMode.EXPORT:
            try:
                if sample_inputs is None:
                    raise ValueError("sample_inputs required for EXPORT mode")
                # PyTorch 2.0+ torch.export
                if hasattr(torch, 'export'):
                    exported = torch.export.export(model, tuple(sample_inputs.values()))
                    self._traced_graph = exported.graph
                    return exported
            except Exception as e:
                logger.warning(f"torch.export failed: {e}")
                
        if self.config.mode == ImportMode.SCRIPT:
            try:
                scripted = torch.jit.script(model)
                self._traced_graph = scripted.graph
                return scripted
            except Exception as e:
                logger.warning(f"TorchScript failed, falling back to trace: {e}")
                
        # Fallback: torch.jit.trace
        if sample_inputs is None:
            # Create dummy inputs
            batch = self.config.batch_size or 1
            seq = self.config.seq_length or 512
            vocab = self._model_config.get('vocab_size', 32000)
            import torch
            sample_inputs = {
                "input_ids": torch.randint(0, vocab, (batch, seq)),
            }
            
        try:
            traced = torch.jit.trace(model, tuple(sample_inputs.values()))
            self._traced_graph = traced.graph
            return traced
        except Exception as e:
            raise ValueError(f"Failed to trace model: {e}")
    
    def _extract_model_config(self, model) -> Dict[str, Any]:
        """Extract model configuration from a PyTorch model.
        
        Note: For HuggingFace models, this requires the `transformers` package.
        Install with: pip install transformers
        """
        config = {}
        
        # Try to get config from HuggingFace model
        if hasattr(model, 'config'):
            hf_config = model.config
            config['num_layers'] = getattr(hf_config, 'num_hidden_layers', 32)
            config['hidden_size'] = getattr(hf_config, 'hidden_size', 4096)
            config['num_attention_heads'] = getattr(hf_config, 'num_attention_heads', 32)
            config['num_key_value_heads'] = getattr(hf_config, 'num_key_value_heads', 
                                                     config['num_attention_heads'])
            config['head_dim'] = config['hidden_size'] // config['num_attention_heads']
            config['vocab_size'] = getattr(hf_config, 'vocab_size', 32000)
            config['max_position_embeddings'] = getattr(hf_config, 'max_position_embeddings', 4096)
            
        return config
    
    def get_detected_patterns(self) -> Dict[str, List]:
        """Get detected patterns after import.
        
        Returns:
            Dictionary with 'attention' and 'linear' pattern lists
        """
        return {
            'attention': self.pattern_matcher._attention_patterns,
            'linear': self.pattern_matcher._linear_patterns
        }


def import_pytorch_model(
    model,
    config: Optional[ImportConfig] = None,
    sample_inputs: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to import a PyTorch model.
    
    Args:
        model: PyTorch model to import
        config: Import configuration
        sample_inputs: Sample inputs for tracing
        
    Returns:
        MLIR module as string
        
    Example:
        >>> import torch
        >>> # Requires: pip install transformers
        >>> from transformers import AutoModelForCausalLM
        >>> from llmir.importers.pytorch import import_pytorch_model, ImportConfig
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> config = ImportConfig(
        ...     enable_kv_cache=True,
        ...     max_seq_length=8192,
        ...     dtype="fp16"
        ... )
        >>> mlir = import_pytorch_model(model, config)
    """
    importer = PyTorchImporter(config)
    return importer.import_model(model, sample_inputs)


__all__ = [
    'ImportConfig',
    'ImportMode', 
    'PyTorchImporter',
    'PatternMatcher',
    'AttentionPattern',
    'LinearPattern',
    'import_pytorch_model'
]
