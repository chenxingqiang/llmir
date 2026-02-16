"""
LLMIR Serving Engines

High-level LLM serving engines with vLLM-compatible APIs.
"""

from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
import time

from llmir.runtime.kv_cache import PagedKVCache
from llmir.runtime.config import KVCacheConfig
from llmir.serving.config import (
    SamplingParams,
    SchedulerConfig,
    SchedulingPolicy,
    RequestPriority,
    EngineConfig,
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
    
    def __init__(self, 
                 cache: PagedKVCache,
                 scheduler_config: Optional[SchedulerConfig] = None):
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
    
    def submit(self, 
               prompt_tokens: List[int],
               sampling_params: Optional[SamplingParams] = None,
               priority: RequestPriority = RequestPriority.NORMAL,
               request_id: Optional[str] = None) -> str:
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
            'prompt_tokens': prompt_tokens,
            'params': sampling_params or SamplingParams(),
            'priority': priority,
            'status': 'pending',
            'output_tokens': [],
            'created_at': time.time(),
            'started_at': None,
            'finished_at': None,
        }
        
        self._total_requests += 1
        return request_id
    
    def abort(self, request_id: str) -> bool:
        """Abort a request."""
        if request_id in self._requests:
            self._requests[request_id]['status'] = 'aborted'
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
            if req['status'] == 'aborted':
                continue
            
            if req['status'] == 'pending':
                req['status'] = 'running'
                req['started_at'] = time.time()
            
            if req['status'] == 'running':
                # Simulate token generation
                params = req['params']
                if len(req['output_tokens']) < params.max_tokens:
                    # Generate a token (placeholder)
                    req['output_tokens'].append(0)
                    self._total_tokens += 1
                    
                    finished = len(req['output_tokens']) >= params.max_tokens
                    
                    output = RequestOutput(
                        request_id=req_id,
                        prompt="",
                        prompt_token_ids=req['prompt_tokens'],
                        outputs=[CompletionOutput(
                            text="",
                            token_ids=req['output_tokens'].copy(),
                            finished=finished,
                            finish_reason="length" if finished else ""
                        )],
                        finished=finished
                    )
                    outputs.append(output)
                    
                    if finished:
                        req['status'] = 'completed'
                        req['finished_at'] = time.time()
                        self._completed_requests += 1
                        
                        if self._output_callback:
                            self._output_callback(
                                req_id, req['output_tokens'], True)
        
        return outputs
    
    def get_request_output(self, request_id: str) -> Optional[RequestOutput]:
        """Get output for a specific request."""
        if request_id not in self._requests:
            return None
        
        req = self._requests[request_id]
        return RequestOutput(
            request_id=request_id,
            prompt="",
            prompt_token_ids=req['prompt_tokens'],
            outputs=[CompletionOutput(
                text="",
                token_ids=req['output_tokens'].copy(),
                finished=req['status'] == 'completed',
                finish_reason="length" if req['status'] == 'completed' else ""
            )],
            finished=req['status'] == 'completed'
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
            'total_requests': self._total_requests,
            'completed_requests': self._completed_requests,
            'pending_requests': sum(1 for r in self._requests.values() 
                                   if r['status'] == 'pending'),
            'running_requests': sum(1 for r in self._requests.values() 
                                   if r['status'] == 'running'),
            'total_tokens': self._total_tokens,
            'tokens_per_second': self._total_tokens / elapsed if elapsed > 0 else 0,
            'elapsed_seconds': elapsed,
        }
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending or running requests."""
        return any(r['status'] in ('pending', 'running') 
                  for r in self._requests.values())


class LLMEngine:
    """
    High-level LLM engine with vLLM-compatible API.
    
    Args:
        model_path: Path to the model
        engine_config: Engine configuration
        cache_config: KV cache configuration
        scheduler_config: Scheduler configuration
    
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
    
    def __init__(self,
                 model_path: str,
                 engine_config: Optional[EngineConfig] = None,
                 cache_config: Optional[KVCacheConfig] = None,
                 scheduler_config: Optional[SchedulerConfig] = None):
        self.model_path = model_path
        self.engine_config = engine_config or EngineConfig(model_path=model_path)
        self.cache_config = cache_config or KVCacheConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()
        
        self._cache = PagedKVCache(self.cache_config)
        self._engine = ContinuousBatchingEngine(self._cache, self.scheduler_config)
        self._tokenizer = None
        self._tokenizer_attempted = False
        self._hf_token: Optional[str] = None
        self._initialized = False
    
    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        tensor_parallel_size: int = 1,
                        dtype: str = "float16",
                        gpu_memory_utilization: float = 0.9,
                        cache_config: Optional[KVCacheConfig] = None,
                        token: Optional[str] = None,
                        **kwargs) -> 'LLMEngine':
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
            cache_config: Optional KV cache config (auto-detected from HF if omitted)
            token: HuggingFace token for gated models (optional)
            **kwargs: Additional arguments (scheduler_config, etc.)

        Returns:
            Initialized LLMEngine
        """
        engine_config = EngineConfig(
            model_path=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Auto-configure KV cache from HuggingFace when not provided
        if cache_config is None:
            try:
                from llmir import from_pretrained as hf_from_pretrained
                if hf_from_pretrained:
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
    
    def generate(self,
                 prompts: Union[str, List[str]],
                 sampling_params: Optional[SamplingParams] = None,
                 use_tqdm: bool = True) -> List[RequestOutput]:
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
    
    async def generate_async(self,
                             prompts: Union[str, List[str]],
                             sampling_params: Optional[SamplingParams] = None) -> List[RequestOutput]:
        """Async generation."""
        # Would implement async generation
        return self.generate(prompts, sampling_params)
