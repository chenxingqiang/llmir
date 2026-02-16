"""
HuggingFace Transformers integration for LLMIR.

Loads model configuration from HuggingFace Hub and creates
LLMIR ModelOptimizer for KV cache and inference optimization.
Supports all decoder-only transformer architectures in the
transformers package.
"""

from typing import Optional, Union, Any, Tuple

from llmir.models import (
    ModelConfig,
    ModelOptimizer,
    ModelArchitecture,
    AttentionType,
    LlamaOptimizer,
    MistralOptimizer,
    PhiOptimizer,
    QwenOptimizer,
    GemmaOptimizer,
    FalconOptimizer,
)

__all__ = ["from_pretrained"]


def _getattr_safe(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute, supporting both attribute and dict access."""
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return default


# model_type strings (from HuggingFace config) -> ModelArchitecture
# Covers decoder-only LLMs in transformers; non-LM or encoder-only use CUSTOM
_MODEL_TYPE_TO_ARCH: Tuple[Tuple[str, ModelArchitecture], ...] = (
    # Llama family (model_type "llama" used for both L2 and L3; Llama 3 optimizer handles both)
    ("llama", ModelArchitecture.LLAMA3),
    ("llama3", ModelArchitecture.LLAMA3),
    ("llama4", ModelArchitecture.LLAMA3),
    ("code_llama", ModelArchitecture.LLAMA2),
    ("diffllama", ModelArchitecture.LLAMA2),
    # Mistral / Mixtral
    ("mistral", ModelArchitecture.MISTRAL),
    ("mistral3", ModelArchitecture.MISTRAL),
    ("ministral", ModelArchitecture.MISTRAL),
    ("ministral3", ModelArchitecture.MISTRAL),
    ("mixtral", ModelArchitecture.MIXTRAL),
    ("voxtral", ModelArchitecture.MIXTRAL),
    # Phi
    ("phi", ModelArchitecture.PHI),
    ("phi3", ModelArchitecture.PHI3),
    ("phimoe", ModelArchitecture.PHI3),
    ("phi4_multimodal", ModelArchitecture.PHI3),
    # Qwen
    ("qwen", ModelArchitecture.QWEN2),
    ("qwen2", ModelArchitecture.QWEN2),
    ("qwen2_moe", ModelArchitecture.QWEN2),
    ("qwen2_vl", ModelArchitecture.QWEN2),
    ("qwen2_audio", ModelArchitecture.QWEN2),
    ("qwen2_5", ModelArchitecture.QWEN2),
    ("qwen2_5_vl", ModelArchitecture.QWEN2),
    ("qwen2_5_omni", ModelArchitecture.QWEN2),
    ("qwen3", ModelArchitecture.QWEN2),
    ("qwen3_moe", ModelArchitecture.QWEN2),
    ("qwen3_5", ModelArchitecture.QWEN2),
    ("qwen3_5_moe", ModelArchitecture.QWEN2),
    ("qwen3_vl", ModelArchitecture.QWEN2),
    ("qwen3_next", ModelArchitecture.QWEN2),
    ("qwen3_omni_moe", ModelArchitecture.QWEN2),
    ("colqwen2", ModelArchitecture.QWEN2),
    # Gemma
    ("gemma", ModelArchitecture.GEMMA),
    ("gemma2", ModelArchitecture.GEMMA),
    ("gemma3", ModelArchitecture.GEMMA),
    ("gemma3_text", ModelArchitecture.GEMMA),
    ("gemma3n", ModelArchitecture.GEMMA),
    ("gemma3n_text", ModelArchitecture.GEMMA),
    ("recurrent_gemma", ModelArchitecture.GEMMA),
    ("t5gemma", ModelArchitecture.GEMMA),
    ("t5gemma2", ModelArchitecture.GEMMA),
    ("vaultgemma", ModelArchitecture.GEMMA),
    ("shieldgemma2", ModelArchitecture.GEMMA),
    # Falcon
    ("falcon", ModelArchitecture.FALCON),
    ("falcon_h1", ModelArchitecture.FALCON),
    ("falcon_mamba", ModelArchitecture.FALCON),
)


def _architecture_from_model_type(model_type: str) -> ModelArchitecture:
    """Map HuggingFace model_type to ModelArchitecture."""
    mt = (model_type or "").lower()
    for prefix, arch in _MODEL_TYPE_TO_ARCH:
        if prefix in mt or mt == prefix:
            return arch
    return ModelArchitecture.CUSTOM


def _model_config_from_hf(hf_config: Any) -> ModelConfig:
    """Map HuggingFace PretrainedConfig to LLMIR ModelConfig.

    Supports attribute names used across all transformer architectures
    (num_hidden_layers/n_layer, hidden_size/n_embd/d_model, etc.).
    """
    model_type = _getattr_safe(hf_config, "model_type")
    if not model_type and getattr(hf_config, "architectures", None):
        archs = getattr(hf_config, "architectures")
        model_type = archs[0] if archs else None
    model_type = (model_type or "").lower()

    # Layers
    num_layers = (
        _getattr_safe(hf_config, "num_hidden_layers")
        or _getattr_safe(hf_config, "n_layer")
        or _getattr_safe(hf_config, "num_layers")
        or 32
    )

    # Hidden size (n_embd=GPT2, d_model=T5-style, decoder_width=some VL)
    hidden_size = (
        _getattr_safe(hf_config, "hidden_size")
        or _getattr_safe(hf_config, "n_embd")
        or _getattr_safe(hf_config, "d_model")
        or _getattr_safe(hf_config, "decoder_width")
        or 4096
    )

    # Attention heads
    num_attention_heads = (
        _getattr_safe(hf_config, "num_attention_heads")
        or _getattr_safe(hf_config, "n_head")
        or _getattr_safe(hf_config, "num_heads")
        or 32
    )

    # KV heads (num_kv_heads=Falcon, num_key_value_heads=Llama-style, n_kv_heads=short)
    num_key_value_heads = (
        _getattr_safe(hf_config, "num_key_value_heads")
        or _getattr_safe(hf_config, "num_kv_heads")
        or _getattr_safe(hf_config, "n_kv_heads")
    )
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    # Head dimension
    head_dim = _getattr_safe(hf_config, "head_dim")
    if head_dim is None or head_dim <= 0:
        head_dim = hidden_size // num_attention_heads if num_attention_heads else 128

    # Intermediate / FFN size (ffn_hidden_size=Falcon, n_inner=GPT2-style)
    intermediate_size = (
        _getattr_safe(hf_config, "intermediate_size")
        or _getattr_safe(hf_config, "n_inner")
        or _getattr_safe(hf_config, "ffn_hidden_size")
        or _getattr_safe(hf_config, "ffn_dim")
    )
    if intermediate_size is None or intermediate_size <= 0:
        intermediate_size = 4 * hidden_size

    # Vocab
    vocab_size = (
        _getattr_safe(hf_config, "vocab_size")
        or _getattr_safe(hf_config, "n_vocab")
        or _getattr_safe(hf_config, "padded_vocab_size")
        or 32000
    )

    # Max position (n_positions=GPT2, n_ctx=some, model_max_length=some)
    max_position_embeddings = (
        _getattr_safe(hf_config, "max_position_embeddings")
        or _getattr_safe(hf_config, "n_ctx")
        or _getattr_safe(hf_config, "n_positions")
        or _getattr_safe(hf_config, "model_max_length")
        or 4096
    )

    # RoPE
    rope_theta = (
        _getattr_safe(hf_config, "rope_theta")
        or _getattr_safe(hf_config, "rotary_emb_base")
        or _getattr_safe(hf_config, "rotary_embedding_base")
        or 10000.0
    )
    rope_scaling = _getattr_safe(hf_config, "rope_scaling")
    rope_scaling_factor = 1.0
    if rope_scaling is not None:
        if isinstance(rope_scaling, dict):
            rope_scaling_factor = rope_scaling.get("factor", 1.0)
        elif hasattr(rope_scaling, "factor"):
            rope_scaling_factor = getattr(rope_scaling, "factor", 1.0)

    # Sliding window
    sliding_window_size = _getattr_safe(hf_config, "sliding_window") or _getattr_safe(hf_config, "sliding_window_size")
    if sliding_window_size is None:
        sliding_window_size = max_position_embeddings

    # Attention type
    attention_type = AttentionType.MULTI_HEAD
    if num_key_value_heads == 1:
        attention_type = AttentionType.MULTI_QUERY
    elif num_key_value_heads and num_key_value_heads < num_attention_heads:
        attention_type = AttentionType.GROUPED_QUERY
    if sliding_window_size and sliding_window_size < max_position_embeddings:
        attention_type = AttentionType.SLIDING_WINDOW

    architecture = _architecture_from_model_type(model_type)

    return ModelConfig(
        architecture=architecture,
        attention_type=attention_type,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        rope_scaling_factor=rope_scaling_factor,
        sliding_window_size=sliding_window_size,
    )


def _create_optimizer(config: ModelConfig) -> ModelOptimizer:
    """Create the appropriate optimizer subclass from ModelConfig."""
    arch = config.architecture
    if arch in (ModelArchitecture.LLAMA, ModelArchitecture.LLAMA2, ModelArchitecture.LLAMA3):
        return LlamaOptimizer(config)
    if arch in (ModelArchitecture.MISTRAL, ModelArchitecture.MIXTRAL):
        return MistralOptimizer(config)
    if arch in (ModelArchitecture.PHI, ModelArchitecture.PHI3):
        return PhiOptimizer(config)
    if arch == ModelArchitecture.QWEN2:
        return QwenOptimizer(config)
    if arch == ModelArchitecture.GEMMA:
        return GemmaOptimizer(config)
    if arch == ModelArchitecture.FALCON:
        return FalconOptimizer(config)
    return ModelOptimizer(config)


def from_pretrained(
    model_id: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
) -> ModelOptimizer:
    """
    Load LLMIR ModelOptimizer from a HuggingFace model identifier.

    Downloads only the model configuration (config.json), not weights.
    Requires the ``transformers`` package (install with ``pip install llmir[full]``).

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct").
        token: HF token for gated models (optional).
        revision: Git revision (branch/tag/commit) for the model (optional).
        trust_remote_code: Allow loading custom code from the Hub (optional).

    Returns:
        ModelOptimizer configured for the given model. Use
        ``optimizer.get_optimized_kv_cache_config()`` to get KVCacheConfig.

    Raises:
        ImportError: If transformers is not installed.
        Exception: If the model config cannot be loaded from HuggingFace Hub.

    Example:
        >>> optimizer = from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        >>> kv_config = optimizer.get_optimized_kv_cache_config()
        >>> print(kv_config.num_layers, kv_config.num_heads)
    """
    try:
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError(
            "HuggingFace integration requires transformers. "
            "Install with: pip install llmir[full]"
        ) from e

    kwargs: dict = {"trust_remote_code": trust_remote_code}
    if token is not None:
        kwargs["token"] = token
    if revision is not None:
        kwargs["revision"] = revision

    hf_config = AutoConfig.from_pretrained(model_id, **kwargs)
    llmir_config = _model_config_from_hf(hf_config)
    return _create_optimizer(llmir_config)
