"""
LLMIR paged-KV-cache decoder.

Provides a manual greedy/sampling decode loop where every layer's K/V tensors
produced by a HuggingFace ``transformers`` model are routed through LLMIR's
:class:`llmir.runtime.PagedKVCache` between steps. This is the kernel-layer
integration point: the model executor runs the forward pass, but KV state is
owned and managed by LLMIR rather than by the model framework's default
``DynamicCache`` or by an external engine like vLLM.

The same decode pattern was first prototyped in ``scripts/mps_full_pipeline.py``.
This module generalizes it so it can drive any HF causal LM on CPU/MPS/CUDA
and is reused by ``LLMEngine`` for the ``BackendType.LLMIR_PAGED`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache

__all__ = ["DecodeResult", "PagedKVDecoder", "kv_config_from_hf_config"]


@dataclass
class DecodeResult:
    """Result of a single prompt's paged decode loop.

    Attributes:
        prompt_token_ids: Token IDs of the input prompt (after tokenization).
        generated_token_ids: Token IDs produced by the decode loop, in order.
        text: Decoded string for ``generated_token_ids``.
        finish_reason: ``"length"`` if ``max_tokens`` was reached or
            ``"stop"`` if the EOS / a stop token was emitted.
    """

    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    text: str
    finish_reason: str


def kv_config_from_hf_config(
    hf_config: Any, *, dtype: str = "float32"
) -> KVCacheConfig:
    """Build a :class:`KVCacheConfig` from a HuggingFace model config.

    Falls back to conservative defaults for fields that are not present.
    """

    num_layers = (
        getattr(hf_config, "num_hidden_layers", None)
        or getattr(hf_config, "n_layer", None)
        or getattr(hf_config, "num_layers", None)
        or 1
    )
    num_kv_heads = (
        getattr(hf_config, "num_key_value_heads", None)
        or getattr(hf_config, "num_attention_heads", None)
        or getattr(hf_config, "n_head", None)
        or 1
    )
    hidden_size = (
        getattr(hf_config, "hidden_size", None)
        or getattr(hf_config, "n_embd", None)
        or 0
    )
    num_attn_heads = (
        getattr(hf_config, "num_attention_heads", None)
        or getattr(hf_config, "n_head", None)
        or num_kv_heads
    )
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None and hidden_size and num_attn_heads:
        head_dim = hidden_size // num_attn_heads
    if not head_dim:
        head_dim = 64
    max_seq_len = (
        getattr(hf_config, "max_position_embeddings", None)
        or getattr(hf_config, "n_positions", None)
        or 2048
    )
    return KVCacheConfig(
        num_layers=int(num_layers),
        num_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        max_seq_len=int(max_seq_len),
        dtype=dtype,
        enable_gpu=False,
    )


class PagedKVDecoder:
    """Drive a HuggingFace causal LM with LLMIR's :class:`PagedKVCache` in the loop.

    For each generation step:

    1. Run ``model(...)`` with either no cache (prefill) or a
       ``DynamicCache`` reconstructed from LLMIR's :class:`PagedKVCache`
       (decode).
    2. Append the model's freshly produced K/V slice for that step into the
       per-layer ``PagedKVCache`` instances. For prefill this is the full
       prompt-length KV; for decode it is just the last token's KV.
    3. Greedy-sample the next token from ``logits[:, -1]`` and append it to
       the running output.

    The point of routing K/V through LLMIR is not (yet) to make the kernel
    *faster* — ``PagedKVCache`` is a numpy-backed reference implementation —
    but to put LLMIR-owned data structures on the critical path so that
    further optimizations (block-paged storage, quantization, prefix sharing,
    speculative branches) actually take effect end-to-end. This is what
    distinguishes the ``LLMIR_PAGED`` backend from the pass-through ``VLLM``
    one.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kv_config: Optional[KVCacheConfig] = None,
        *,
        device: Any = None,
        dtype: Any = None,
    ):
        # Imported lazily to keep the module import cheap and to make the
        # transformers / torch dependency explicit at construction time.
        import torch  # noqa: F401  (validated availability)

        self.model = model
        self.tokenizer = tokenizer
        self._device = device or self._detect_device()
        self._dtype = dtype or self._detect_dtype()
        self.kv_config = kv_config or kv_config_from_hf_config(
            getattr(model, "config", None) or type("_C", (), {})(),
            dtype=str(self._dtype).replace("torch.", ""),
        )
        # Number of decoder layers in the underlying model. We use it to
        # validate that the inferred ``kv_config.num_layers`` is consistent.
        self._num_layers = self._infer_num_layers()
        if self._num_layers and self.kv_config.num_layers != self._num_layers:
            # Trust the model — KVCache must match its layer count.
            self.kv_config.num_layers = self._num_layers

    # ------------------------------------------------------------------ utils

    def _detect_device(self) -> Any:
        import torch

        # Prefer the model's actual device when discoverable.
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _detect_dtype(self) -> Any:
        import torch

        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _infer_num_layers(self) -> int:
        config = getattr(self.model, "config", None)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(config, attr, None)
            if value:
                return int(value)
        return 0

    # ----------------------------------------------------------- public API

    def decode(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> List[DecodeResult]:
        """Run the paged decode loop for each prompt.

        Each prompt is decoded independently with its own per-layer cache;
        this keeps the implementation simple and matches ``temperature=0.0``
        greedy semantics used by the CPU benchmark. Batched prefill could be
        added later without changing the public surface.
        """

        results: List[DecodeResult] = []
        eos = eos_token_id
        if eos is None and self.tokenizer is not None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        stops = {int(t) for t in (stop_token_ids or ())}
        if eos is not None:
            stops.add(int(eos))

        for prompt in prompts:
            results.append(
                self._decode_one(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop_ids=stops,
                )
            )
        return results

    # ---------------------------------------------------------------- impl

    def _decode_one(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_ids: set,
    ) -> DecodeResult:
        import torch

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.kv_config.max_seq_len, 4096),
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        else:
            attention_mask = torch.ones_like(
                input_ids, dtype=torch.long, device=self._device
            )

        prompt_token_ids = [int(t) for t in input_ids[0].tolist()]
        prompt_len = input_ids.shape[1]

        # One PagedKVCache per layer. Allocating per request keeps the API
        # simple and mirrors how a real engine would scope cache lifetimes
        # to a sequence (or set of sequences).
        layer_caches: List[PagedKVCache] = [
            PagedKVCache(self.kv_config) for _ in range(self.kv_config.num_layers)
        ]
        seq_ids = np.zeros(input_ids.shape[0], dtype=np.int32)

        cache_position = torch.arange(
            prompt_len, dtype=torch.int64, device=self._device
        )
        next_input_ids = input_ids
        generated: List[int] = []
        finish_reason = "length"

        first_step = True
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if first_step:
                    model_inputs = {
                        "input_ids": next_input_ids,
                        "attention_mask": attention_mask,
                        "cache_position": cache_position,
                        "use_cache": True,
                    }
                else:
                    past_len = layer_caches[0].get_sequence_length(0)
                    past_key_values = self._lookup_dynamic_cache(layer_caches, past_len)
                    model_inputs = {
                        "input_ids": next_input_ids,
                        "attention_mask": attention_mask,
                        "past_key_values": past_key_values,
                        "cache_position": cache_position,
                        "use_cache": True,
                    }

                outputs = self.model(**model_inputs)

                # Capture this step's K/V into LLMIR PagedKVCache. On the
                # first (prefill) step the model returns the full prompt-length
                # KV; on later (decode) steps we want only the freshly added
                # tail token, so we slice.
                self._append_to_paged(
                    layer_caches,
                    getattr(outputs, "past_key_values", None),
                    seq_ids,
                    append_only_new=not first_step,
                )

                logits = outputs.logits
                next_token = int(logits[:, -1].argmax(dim=-1).item())
                generated.append(next_token)

                if next_token in stop_ids:
                    finish_reason = "stop"
                    break

                # Prepare next iteration: feed only the new token, extend the
                # attention mask, and bump cache_position by one.
                next_input_ids = torch.tensor(
                    [[next_token]], dtype=input_ids.dtype, device=self._device
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
                cache_position = cache_position[-1:] + 1
                first_step = False

        text = (
            self.tokenizer.decode(generated, skip_special_tokens=True)
            if generated
            else ""
        )
        return DecodeResult(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated,
            text=text,
            finish_reason=finish_reason,
        )

    # --------------------------------------------------------- KV plumbing

    def _append_to_paged(
        self,
        layer_caches: List[PagedKVCache],
        past_key_values: Any,
        seq_ids: np.ndarray,
        *,
        append_only_new: bool,
    ) -> None:
        """Append per-layer K/V from the model's cache into LLMIR's PagedKVCache."""

        layers = self._iter_layers(past_key_values)
        for layer_idx, (k, v) in enumerate(layers):
            if layer_idx >= len(layer_caches):
                break
            # transformers' DynamicCache uses (batch, num_heads, seq_len, head_dim).
            # When decoding step-by-step we only want the tail slice.
            if append_only_new and k.shape[2] > 1:
                k = k[:, :, -1:, :]
                v = v[:, :, -1:, :]
            # PagedKVCache.append expects (batch, seq_len, num_heads, head_dim).
            k_np = k.detach().to("cpu").float().numpy().transpose(0, 2, 1, 3)
            v_np = v.detach().to("cpu").float().numpy().transpose(0, 2, 1, 3)
            layer_caches[layer_idx].append(k_np, v_np, seq_ids)

    def _lookup_dynamic_cache(
        self,
        layer_caches: List[PagedKVCache],
        past_len: int,
    ) -> Any:
        """Reconstruct a transformers DynamicCache from LLMIR PagedKVCache state."""

        import torch
        from transformers import DynamicCache

        batch_size = 1
        block_indices = np.zeros(
            (batch_size, self.kv_config.num_layers), dtype=np.int32
        )
        seq_lens = np.full(batch_size, past_len, dtype=np.int32)
        layer_data: List[Tuple[Any, Any]] = []
        for lc in layer_caches:
            k_np, v_np = lc.lookup(block_indices, seq_lens)
            # PagedKVCache returns (batch, seq_len, num_heads, head_dim);
            # transformers wants (batch, num_heads, seq_len, head_dim).
            k_t = torch.from_numpy(k_np.transpose(0, 2, 1, 3)).to(
                device=self._device, dtype=self._dtype
            )
            v_t = torch.from_numpy(v_np.transpose(0, 2, 1, 3)).to(
                device=self._device, dtype=self._dtype
            )
            layer_data.append((k_t, v_t))
        # transformers >= 4.40 supports the kwarg constructor; older versions
        # expose ``key_cache`` / ``value_cache`` lists. Try both.
        try:
            return DynamicCache(_distributed_cache_data=layer_data)
        except TypeError:
            cache = DynamicCache()
            for layer_idx, (k_t, v_t) in enumerate(layer_data):
                cache.update(k_t, v_t, layer_idx)
            return cache

    @staticmethod
    def _iter_layers(past_key_values: Any) -> List[Tuple[Any, Any]]:
        """Normalize transformers' various cache shapes into a list of (k, v)."""

        if past_key_values is None:
            return []
        # Newer transformers: DynamicCache exposes key_cache / value_cache lists.
        keys = getattr(past_key_values, "key_cache", None)
        values = getattr(past_key_values, "value_cache", None)
        if keys is not None and values is not None:
            return list(zip(keys, values))
        # Legacy tuple-of-tuples form: ((k0, v0), (k1, v1), ...).
        try:
            return [(layer[0], layer[1]) for layer in past_key_values]
        except (TypeError, IndexError):
            return []
