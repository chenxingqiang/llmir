"""
LLMIR Serving Engines

High-level LLM serving engines with vLLM-compatible APIs.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from llmir.runtime.config import KVCacheConfig
from llmir.runtime.kv_cache import PagedKVCache
from llmir.serving.config import (
    BackendType,
    EngineConfig,
    RequestPriority,
    SamplingParams,
    SchedulerConfig,
)


@dataclass
class CompletionOutput:
    """Output from a single completion."""

    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: str = ""
    logprobs: Optional[List[float]] = None
    cumulative_logprob: float = 0.0


@dataclass
class RequestOutput:
    """Output from a request."""

    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Optional[Dict[str, float]] = None


def _normalize_backend(backend: Union[BackendType, str]) -> str:
    """Return a supported backend name."""
    backend_name = backend.value if isinstance(backend, BackendType) else str(backend)
    backend_name = backend_name.lower()
    if backend_name not in {
        BackendType.LLMIR.value,
        BackendType.VLLM.value,
        BackendType.LLMIR_PAGED.value,
    }:
        raise ValueError(f"Unsupported backend: {backend}")
    return backend_name


class ContinuousBatchingEngine:
    """
    Continuous batching engine for production LLM serving.

    Implements vLLM-style dynamic batch management with preemption support.

    Args:
        cache: PagedKVCache instance
        scheduler_config: Scheduler configuration

    Example:
        >>> config = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)
        >>> cache = PagedKVCache(config)
        >>> scheduler_config = SchedulerConfig(policy=SchedulingPolicy.ADAPTIVE)
        >>>
        >>> engine = ContinuousBatchingEngine(cache, scheduler_config)
        >>> engine.start()
        >>>
        >>> # Submit requests
        >>> request_id = engine.submit([1, 2, 3, 4], SamplingParams(max_tokens=100))
        >>>
        >>> # Get outputs
        >>> for output in engine.iterate():
        ...     print(output.outputs[0].text)
    """

    def __init__(
        self, cache: PagedKVCache, scheduler_config: Optional[SchedulerConfig] = None
    ):
        self.cache = cache
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self._running = False
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0
        self._output_callback: Optional[Callable] = None
        self._error_callback: Optional[Callable] = None

        # Statistics
        self._total_requests = 0
        self._completed_requests = 0
        self._total_tokens = 0
        self._start_time = 0.0

    def start(self):
        """Start the engine."""
        self._running = True
        self._start_time = time.time()

    def stop(self):
        """Stop the engine."""
        self._running = False

    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    def submit(
        self,
        prompt_tokens: List[int],
        sampling_params: Optional[SamplingParams] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Submit a generation request.

        Args:
            prompt_tokens: Tokenized prompt
            sampling_params: Sampling parameters
            priority: Request priority
            request_id: Optional custom request ID

        Returns:
            request_id: Unique identifier for the request
        """
        if request_id is None:
            request_id = f"req-{self._next_id}"
            self._next_id += 1

        self._requests[request_id] = {
            "prompt_tokens": prompt_tokens,
            "params": sampling_params or SamplingParams(),
            "priority": priority,
            "status": "pending",
            "output_tokens": [],
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
        }

        self._total_requests += 1
        return request_id

    def abort(self, request_id: str) -> bool:
        """Abort a request."""
        if request_id in self._requests:
            self._requests[request_id]["status"] = "aborted"
            return True
        return False

    def iterate(self) -> List[RequestOutput]:
        """
        Run one iteration and return outputs.

        Returns:
            List of RequestOutput for completed/updated requests
        """
        outputs = []

        for req_id, req in list(self._requests.items()):
            if req["status"] == "aborted":
                continue

            if req["status"] == "pending":
                req["status"] = "running"
                req["started_at"] = time.time()

            if req["status"] == "running":
                # Simulate token generation
                params = req["params"]
                if len(req["output_tokens"]) < params.max_tokens:
                    # Generate a token (placeholder)
                    req["output_tokens"].append(0)
                    self._total_tokens += 1

                    finished = len(req["output_tokens"]) >= params.max_tokens

                    output = RequestOutput(
                        request_id=req_id,
                        prompt="",
                        prompt_token_ids=req["prompt_tokens"],
                        outputs=[
                            CompletionOutput(
                                text="",
                                token_ids=req["output_tokens"].copy(),
                                finished=finished,
                                finish_reason="length" if finished else "",
                            )
                        ],
                        finished=finished,
                    )
                    outputs.append(output)

                    if finished:
                        req["status"] = "completed"
                        req["finished_at"] = time.time()
                        self._completed_requests += 1

                        if self._output_callback:
                            self._output_callback(req_id, req["output_tokens"], True)

        return outputs

    def get_request_output(self, request_id: str) -> Optional[RequestOutput]:
        """Get output for a specific request."""
        if request_id not in self._requests:
            return None

        req = self._requests[request_id]
        return RequestOutput(
            request_id=request_id,
            prompt="",
            prompt_token_ids=req["prompt_tokens"],
            outputs=[
                CompletionOutput(
                    text="",
                    token_ids=req["output_tokens"].copy(),
                    finished=req["status"] == "completed",
                    finish_reason="length" if req["status"] == "completed" else "",
                )
            ],
            finished=req["status"] == "completed",
        )

    def set_output_callback(self, callback: Callable):
        """Set callback for output tokens."""
        self._output_callback = callback

    def set_error_callback(self, callback: Callable):
        """Set callback for errors."""
        self._error_callback = callback

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "total_requests": self._total_requests,
            "completed_requests": self._completed_requests,
            "pending_requests": sum(
                1 for r in self._requests.values() if r["status"] == "pending"
            ),
            "running_requests": sum(
                1 for r in self._requests.values() if r["status"] == "running"
            ),
            "total_tokens": self._total_tokens,
            "tokens_per_second": self._total_tokens / elapsed if elapsed > 0 else 0,
            "elapsed_seconds": elapsed,
        }

    def has_pending_requests(self) -> bool:
        """Check if there are pending or running requests."""
        return any(
            r["status"] in ("pending", "running") for r in self._requests.values()
        )


class LLMEngine:
    """
    High-level LLM engine with vLLM-compatible API.

    Args:
        model_path: Path to the model
        engine_config: Optional engine configuration. If omitted, a default
            EngineConfig is created from model_path.
        cache_config: Optional KV cache configuration. If omitted, a default
            KVCacheConfig is used.
        scheduler_config: Optional scheduler configuration. If omitted, a
            default SchedulerConfig is used.
        backend: Optional serving backend override. If omitted,
            engine_config.backend is used.

    Example:
        >>> engine = LLMEngine.from_pretrained("meta-llama/Llama-3.1-8B")
        >>>
        >>> # Generate completions
        >>> outputs = engine.generate(
        ...     ["Hello, how are you?", "What is 2+2?"],
        ...     SamplingParams(max_tokens=100)
        ... )
        >>>
        >>> for output in outputs:
        ...     print(output.outputs[0].text)
    """

    def __init__(
        self,
        model_path: str,
        engine_config: Optional[EngineConfig] = None,
        cache_config: Optional[KVCacheConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        backend: Optional[Union[BackendType, str]] = None,
    ):
        self.model_path = model_path
        self.engine_config = engine_config or EngineConfig(model_path=model_path)
        self.cache_config = cache_config or KVCacheConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()

        self._cache = PagedKVCache(self.cache_config)
        self._engine = ContinuousBatchingEngine(self._cache, self.scheduler_config)
        self._tokenizer = None
        self._tokenizer_attempted = False
        self._hf_token: Optional[str] = None
        self._vllm_engine = None
        self._vllm_sampling_params_cls = None
        # LLMIR_PAGED backend (kernel-layer integration) state
        self._hf_model: Any = None
        self._paged_decoder: Any = None
        self._initialized = False
        self.backend = _normalize_backend(
            backend if backend is not None else self.engine_config.backend
        )
        self.engine_config.backend = self.backend

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        cache_config: Optional[KVCacheConfig] = None,
        token: Optional[str] = None,
        backend: Union[BackendType, str] = BackendType.LLMIR,
        **kwargs,
    ) -> "LLMEngine":
        """
        Load engine from a pretrained model.

        When model_name_or_path is a HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B),
        KV cache config is auto-configured from the model (requires llmir[full]).
        Pass cache_config to override.

        Args:
            model_name_or_path: Model name, registry key, or HuggingFace model ID
            tensor_parallel_size: Number of GPUs
            dtype: Data type
            gpu_memory_utilization: GPU memory utilization
            max_model_len: Maximum model context length
            trust_remote_code: Allow remote model code
            cache_config: Optional KV cache config (auto-detected from HF if omitted)
            token: HuggingFace token for gated models (optional)
            backend: Serving backend to use ("llmir" or "vllm")
            **kwargs: Additional arguments (scheduler_config, etc.)

        Returns:
            Initialized LLMEngine
        """
        backend_name = _normalize_backend(backend)
        engine_config = EngineConfig(
            model_path=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            backend=backend_name,
        )

        # Auto-configure KV cache from HuggingFace when not provided
        if cache_config is None:
            try:
                from llmir import from_pretrained as hf_from_pretrained

                if hf_from_pretrained is not None:
                    optimizer = hf_from_pretrained(model_name_or_path, token=token)
                    cache_config = optimizer.get_optimized_kv_cache_config()
            except (ImportError, Exception):
                pass  # Use default KVCacheConfig

        engine = cls(
            model_name_or_path,
            engine_config,
            cache_config=cache_config,
            scheduler_config=kwargs.pop("scheduler_config", None),
        )
        engine._hf_token = token
        return engine

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """
        Generate completions for prompts.

        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters
            use_tqdm: Whether to show progress bar

        Returns:
            List of RequestOutput
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        params = sampling_params or SamplingParams()

        if self.backend == BackendType.VLLM.value:
            return self._generate_vllm(prompts, params)

        if self.backend == BackendType.LLMIR_PAGED.value:
            return self._generate_llmir_paged(prompts, params)

        # Start engine if not running
        if not self._engine.is_running():
            self._engine.start()

        # Submit all prompts
        request_ids = []
        for prompt in prompts:
            # Tokenize (placeholder - would use actual tokenizer)
            tokens = self._tokenize(prompt)
            req_id = self._engine.submit(tokens, params)
            request_ids.append(req_id)

        # Collect outputs
        outputs = []
        while self._engine.has_pending_requests():
            self._engine.iterate()

        for req_id in request_ids:
            output = self._engine.get_request_output(req_id)
            if output:
                # Detokenize (placeholder)
                for completion in output.outputs:
                    completion.text = self._detokenize(completion.token_ids)
                outputs.append(output)

        return outputs

    def _ensure_vllm(self) -> None:
        """Load and initialize vLLM lazily."""
        if self._vllm_engine is not None:
            return

        try:
            from vllm import LLM
            from vllm import SamplingParams as VLLMSamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM backend requested but vllm is not installed. "
                "Install vLLM separately to use backend='vllm'."
            ) from exc

        vllm_kwargs: Dict[str, Any] = {
            "model": self.model_path,
            "tensor_parallel_size": self.engine_config.tensor_parallel_size,
            "dtype": self.engine_config.dtype,
            "gpu_memory_utilization": self.engine_config.gpu_memory_utilization,
            "trust_remote_code": self.engine_config.trust_remote_code,
        }
        if self.engine_config.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.engine_config.max_model_len

        self._vllm_engine = LLM(**vllm_kwargs)
        self._vllm_sampling_params_cls = VLLMSamplingParams

    def _to_vllm_sampling_params(self, params: SamplingParams) -> Any:
        """Convert LLMIR sampling parameters to vLLM sampling parameters."""
        self._ensure_vllm()
        kwargs = params.to_dict()
        if not kwargs.get("stop"):
            kwargs.pop("stop", None)
        if not kwargs.get("stop_token_ids"):
            kwargs.pop("stop_token_ids", None)
        return self._vllm_sampling_params_cls(**kwargs)

    def _generate_vllm(
        self, prompts: List[str], sampling_params: SamplingParams
    ) -> List[RequestOutput]:
        """Generate completions through vLLM and normalize outputs."""
        self._ensure_vllm()
        vllm_params = self._to_vllm_sampling_params(sampling_params)
        vllm_outputs = self._vllm_engine.generate(prompts, vllm_params)

        outputs = []
        for index, output in enumerate(vllm_outputs):
            prompt = getattr(output, "prompt", prompts[index])
            prompt_token_ids = list(getattr(output, "prompt_token_ids", []))
            completions = []
            for completion in getattr(output, "outputs", []):
                finish_reason = getattr(completion, "finish_reason", "") or ""
                completions.append(
                    CompletionOutput(
                        text=getattr(completion, "text", ""),
                        token_ids=list(getattr(completion, "token_ids", [])),
                        finished=bool(finish_reason),
                        finish_reason=finish_reason,
                        logprobs=getattr(completion, "logprobs", None),
                        cumulative_logprob=getattr(
                            completion, "cumulative_logprob", 0.0
                        ),
                    )
                )
            outputs.append(
                RequestOutput(
                    request_id=str(index),
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    outputs=completions,
                    finished=bool(completions)
                    and all(completion.finished for completion in completions),
                )
            )
        return outputs

    def _ensure_llmir_paged(self) -> None:
        """Load HF transformers model + tokenizer for the LLMIR_PAGED path."""

        if self._paged_decoder is not None:
            return

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - transformers absent
            raise ImportError(
                "backend='llmir_paged' requires the 'transformers' package "
                "(install llmir[full] or pip install transformers torch)."
            ) from exc

        from llmir.runtime.paged_decoder import (
            PagedKVDecoder,
            kv_config_from_hf_config,
        )

        self._ensure_tokenizer()
        if self._tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer for {self.model_path!r}; "
                "LLMIR_PAGED backend requires a working HuggingFace tokenizer."
            )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.engine_config.trust_remote_code,
        }
        if self._hf_token:
            load_kwargs["token"] = self._hf_token
        # Map the engine dtype string to a torch dtype when possible.
        try:
            import torch

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.engine_config.dtype.lower())
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
        except ImportError:
            pass

        model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        model.eval()
        self._hf_model = model

        # Honour the user-provided cache_config when its layer count matches
        # the model; otherwise derive a fresh one from the HF config so the
        # PagedKVCache is sized correctly.
        kv_config = kv_config_from_hf_config(
            getattr(model, "config", None) or type("_C", (), {})(),
            dtype=self.engine_config.dtype,
        )
        if (
            self.cache_config is not None
            and self.cache_config.num_layers == kv_config.num_layers
            and self.cache_config.num_heads == kv_config.num_heads
            and self.cache_config.head_dim == kv_config.head_dim
        ):
            kv_config = self.cache_config

        self._paged_decoder = PagedKVDecoder(
            model,
            self._tokenizer,
            kv_config=kv_config,
        )

    def _generate_llmir_paged(
        self, prompts: List[str], sampling_params: SamplingParams
    ) -> List[RequestOutput]:
        """Run the kernel-integrated decode loop and shape outputs."""

        self._ensure_llmir_paged()
        assert self._paged_decoder is not None  # for type-checkers

        stop_token_ids = list(sampling_params.stop_token_ids or ())
        results = self._paged_decoder.decode(
            prompts,
            max_new_tokens=sampling_params.max_tokens,
            stop_token_ids=stop_token_ids,
        )

        outputs: List[RequestOutput] = []
        for index, (prompt, decoded) in enumerate(zip(prompts, results)):
            completion = CompletionOutput(
                text=decoded.text,
                token_ids=list(decoded.generated_token_ids),
                finished=True,
                finish_reason=decoded.finish_reason,
            )
            outputs.append(
                RequestOutput(
                    request_id=str(index),
                    prompt=prompt,
                    prompt_token_ids=list(decoded.prompt_token_ids),
                    outputs=[completion],
                    finished=True,
                )
            )
        return outputs

    def _ensure_tokenizer(self) -> None:
        """Load HuggingFace tokenizer if available (llmir[full])."""
        if self._tokenizer_attempted:
            return
        self._tokenizer_attempted = True
        try:
            from transformers import AutoTokenizer

            kw: Dict[str, Any] = {}
            if self._hf_token:
                kw["token"] = self._hf_token
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, **kw)
        except (ImportError, Exception):
            pass

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text. Uses HuggingFace tokenizer when available."""
        self._ensure_tokenizer()
        if self._tokenizer is not None:
            return self._tokenizer.encode(text, add_special_tokens=True)
        return list(range(len(text.split())))

    def _detokenize(self, token_ids: List[int]) -> str:
        """Detokenize token IDs. Uses HuggingFace tokenizer when available."""
        self._ensure_tokenizer()
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return f"Generated {len(token_ids)} tokens"

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self._engine.get_stats()

    def shutdown(self):
        """Shutdown the engine."""
        self._engine.stop()


# Async versions for production use
class AsyncLLMEngine(LLMEngine):
    """Async version of LLMEngine for production serving."""

    async def generate_async(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """Async generation."""
        # Would implement async generation
        return self.generate(prompts, sampling_params)
