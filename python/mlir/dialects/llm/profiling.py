"""
Performance profiling tools for LLMIR.

This module provides comprehensive profiling capabilities for
analyzing and optimizing LLM inference performance.
"""

import time
import threading
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import contextlib

__all__ = [
    'Profiler',
    'ProfilerConfig',
    'MemoryProfiler',
    'LatencyProfiler',
    'ThroughputMonitor',
    'ProfileReport',
    'EventType',
    'profile',
    'trace',
]

#===----------------------------------------------------------------------===#
# Configuration
#===----------------------------------------------------------------------===#

class EventType(Enum):
    """Types of profiling events."""
    PREFILL = auto()
    DECODE = auto()
    KV_CACHE_APPEND = auto()
    KV_CACHE_LOOKUP = auto()
    ATTENTION = auto()
    MATMUL = auto()
    MEMORY_ALLOC = auto()
    MEMORY_FREE = auto()
    BATCH_SCHEDULE = auto()
    MODEL_FORWARD = auto()
    TOKENIZE = auto()
    DETOKENIZE = auto()
    CUSTOM = auto()


@dataclass
class ProfilerConfig:
    """Configuration for profiler."""
    enabled: bool = True
    trace_memory: bool = True
    trace_cuda: bool = True
    record_shapes: bool = False
    record_stack: bool = False
    with_flops: bool = False
    export_chrome_trace: bool = False
    output_dir: str = "./profiles"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'trace_memory': self.trace_memory,
            'trace_cuda': self.trace_cuda,
            'record_shapes': self.record_shapes,
            'record_stack': self.record_stack,
            'with_flops': self.with_flops,
            'export_chrome_trace': self.export_chrome_trace,
            'output_dir': self.output_dir,
        }


@dataclass
class ProfileEvent:
    """A single profiling event."""
    name: str
    event_type: EventType
    start_time: float
    end_time: float
    duration_ms: float
    thread_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.event_type.name,
            'start_us': int(self.start_time * 1e6),
            'end_us': int(self.end_time * 1e6),
            'duration_ms': self.duration_ms,
            'tid': self.thread_id,
            'args': self.metadata,
        }
    
    def to_chrome_event(self) -> Dict[str, Any]:
        """Convert to Chrome trace format."""
        return {
            'name': self.name,
            'cat': self.event_type.name,
            'ph': 'X',  # Complete event
            'ts': int(self.start_time * 1e6),
            'dur': int(self.duration_ms * 1e3),
            'tid': self.thread_id,
            'pid': os.getpid(),
            'args': self.metadata,
        }


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    num_allocations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


#===----------------------------------------------------------------------===#
# Core Profiler
#===----------------------------------------------------------------------===#

class Profiler:
    """
    Main profiler for LLMIR operations.
    
    Example:
        >>> profiler = Profiler()
        >>> profiler.start()
        >>> 
        >>> with profiler.trace("attention", EventType.ATTENTION):
        ...     output = model.attention(x)
        >>> 
        >>> profiler.stop()
        >>> report = profiler.get_report()
        >>> report.print_summary()
    """
    
    _instance: Optional['Profiler'] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()
        self._events: List[ProfileEvent] = []
        self._memory_snapshots: List[MemorySnapshot] = []
        self._active = False
        self._start_time = 0.0
        self._end_time = 0.0
        self._event_stack: Dict[int, List[str]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'Profiler':
        """Get singleton profiler instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Profiler()
        return cls._instance
    
    def start(self):
        """Start profiling."""
        self._active = True
        self._start_time = time.perf_counter()
        self._events.clear()
        self._memory_snapshots.clear()
        self._counters.clear()
    
    def stop(self):
        """Stop profiling."""
        self._end_time = time.perf_counter()
        self._active = False
    
    def is_active(self) -> bool:
        """Check if profiling is active."""
        return self._active
    
    @contextlib.contextmanager
    def trace(self, name: str, event_type: EventType = EventType.CUSTOM,
              **metadata):
        """
        Context manager for tracing an operation.
        
        Args:
            name: Name of the operation
            event_type: Type of event
            **metadata: Additional metadata to record
        """
        if not self._active or not self.config.enabled:
            yield
            return
        
        thread_id = threading.get_ident()
        start_time = time.perf_counter()
        
        self._event_stack[thread_id].append(name)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            event = ProfileEvent(
                name=name,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                thread_id=thread_id,
                metadata=metadata
            )
            
            with self._lock:
                self._events.append(event)
                self._counters[name] += 1
            
            self._event_stack[thread_id].pop()
    
    def record_event(self, name: str, event_type: EventType,
                     duration_ms: float, **metadata):
        """Record a single event."""
        if not self._active or not self.config.enabled:
            return
        
        now = time.perf_counter()
        event = ProfileEvent(
            name=name,
            event_type=event_type,
            start_time=now - duration_ms / 1000,
            end_time=now,
            duration_ms=duration_ms,
            thread_id=threading.get_ident(),
            metadata=metadata
        )
        
        with self._lock:
            self._events.append(event)
            self._counters[name] += 1
    
    def record_memory(self, allocated: int, reserved: int, peak: int,
                      num_allocs: int, **metadata):
        """Record a memory snapshot."""
        if not self._active or not self.config.trace_memory:
            return
        
        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=peak,
            num_allocations=num_allocs,
            metadata=metadata
        )
        
        with self._lock:
            self._memory_snapshots.append(snapshot)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)
    
    def get_events(self) -> List[ProfileEvent]:
        """Get all recorded events."""
        return list(self._events)
    
    def get_memory_snapshots(self) -> List[MemorySnapshot]:
        """Get all memory snapshots."""
        return list(self._memory_snapshots)
    
    def get_report(self) -> 'ProfileReport':
        """Generate a profile report."""
        return ProfileReport(
            events=self._events,
            memory_snapshots=self._memory_snapshots,
            counters=dict(self._counters),
            total_time_ms=(self._end_time - self._start_time) * 1000
        )
    
    def export_chrome_trace(self, filepath: str):
        """Export trace in Chrome trace format."""
        chrome_events = [e.to_chrome_event() for e in self._events]
        
        trace_data = {
            'traceEvents': chrome_events,
            'displayTimeUnit': 'ms',
            'metadata': {
                'process_name': 'LLMIR',
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f)
    
    def reset(self):
        """Reset all recorded data."""
        with self._lock:
            self._events.clear()
            self._memory_snapshots.clear()
            self._counters.clear()


#===----------------------------------------------------------------------===#
# Profile Report
#===----------------------------------------------------------------------===#

@dataclass
class EventStats:
    """Statistics for a single event type."""
    name: str
    count: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    percentage: float


class ProfileReport:
    """
    Profile report with analysis and visualization.
    
    Example:
        >>> report = profiler.get_report()
        >>> report.print_summary()
        >>> report.save_json("profile_report.json")
    """
    
    def __init__(self,
                 events: List[ProfileEvent],
                 memory_snapshots: List[MemorySnapshot],
                 counters: Dict[str, int],
                 total_time_ms: float):
        self.events = events
        self.memory_snapshots = memory_snapshots
        self.counters = counters
        self.total_time_ms = total_time_ms
        self._stats: Optional[Dict[str, EventStats]] = None
    
    def get_stats(self) -> Dict[str, EventStats]:
        """Compute statistics for each event type."""
        if self._stats is not None:
            return self._stats
        
        # Group events by name
        events_by_name: Dict[str, List[float]] = defaultdict(list)
        for event in self.events:
            events_by_name[event.name].append(event.duration_ms)
        
        # Compute stats
        self._stats = {}
        for name, durations in events_by_name.items():
            n = len(durations)
            total = sum(durations)
            avg = total / n if n > 0 else 0
            
            # Standard deviation
            variance = sum((d - avg) ** 2 for d in durations) / n if n > 1 else 0
            std = variance ** 0.5
            
            self._stats[name] = EventStats(
                name=name,
                count=n,
                total_ms=total,
                avg_ms=avg,
                min_ms=min(durations) if durations else 0,
                max_ms=max(durations) if durations else 0,
                std_ms=std,
                percentage=(total / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            )
        
        return self._stats
    
    def print_summary(self):
        """Print a summary of the profile."""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("LLMIR Performance Profile Summary")
        print("=" * 80)
        print(f"\nTotal Time: {self.total_time_ms:.2f} ms")
        print(f"Total Events: {len(self.events)}")
        print()
        
        # Sort by total time
        sorted_stats = sorted(stats.values(), key=lambda x: -x.total_ms)
        
        # Table header
        header = f"{'Operation':<30} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>10} {'%':>8}"
        print(header)
        print("-" * len(header))
        
        for stat in sorted_stats:
            print(f"{stat.name:<30} {stat.count:>8} {stat.total_ms:>12.2f} "
                  f"{stat.avg_ms:>10.2f} {stat.percentage:>7.1f}%")
        
        # Memory summary
        if self.memory_snapshots:
            print("\n" + "-" * 40)
            print("Memory Summary")
            print("-" * 40)
            
            max_alloc = max(s.allocated_bytes for s in self.memory_snapshots)
            max_reserved = max(s.reserved_bytes for s in self.memory_snapshots)
            
            print(f"Peak Allocated: {max_alloc / 1e9:.2f} GB")
            print(f"Peak Reserved:  {max_reserved / 1e9:.2f} GB")
        
        print()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        stats = self.get_stats()
        
        return {
            'total_time_ms': self.total_time_ms,
            'num_events': len(self.events),
            'stats': {name: {
                'count': s.count,
                'total_ms': s.total_ms,
                'avg_ms': s.avg_ms,
                'min_ms': s.min_ms,
                'max_ms': s.max_ms,
                'std_ms': s.std_ms,
                'percentage': s.percentage,
            } for name, s in stats.items()},
            'counters': self.counters,
            'memory': [{
                'timestamp': s.timestamp,
                'allocated_mb': s.allocated_bytes / 1e6,
                'reserved_mb': s.reserved_bytes / 1e6,
            } for s in self.memory_snapshots],
        }
    
    def save_json(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_bottlenecks(self, top_k: int = 5) -> List[EventStats]:
        """Get top bottleneck operations."""
        stats = self.get_stats()
        sorted_stats = sorted(stats.values(), key=lambda x: -x.total_ms)
        return sorted_stats[:top_k]


#===----------------------------------------------------------------------===#
# Specialized Profilers
#===----------------------------------------------------------------------===#

class MemoryProfiler:
    """
    Memory-focused profiler for tracking allocations.
    
    Example:
        >>> mem_profiler = MemoryProfiler()
        >>> mem_profiler.start()
        >>> 
        >>> # Run operations
        >>> cache = PagedKVCache(config)
        >>> 
        >>> mem_profiler.stop()
        >>> print(mem_profiler.get_peak_memory() / 1e9, "GB")
    """
    
    def __init__(self, sample_interval_ms: float = 100):
        self.sample_interval = sample_interval_ms / 1000
        self._snapshots: List[MemorySnapshot] = []
        self._active = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start memory profiling."""
        self._active = True
        self._snapshots.clear()
        
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop memory profiling."""
        self._active = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def _sample_loop(self):
        """Background sampling loop."""
        while self._active:
            self._take_snapshot()
            time.sleep(self.sample_interval)
    
    def _take_snapshot(self):
        """Take a memory snapshot."""
        # Would query actual GPU memory here
        # For now, use placeholder values
        snapshot = MemorySnapshot(
            timestamp=time.perf_counter(),
            allocated_bytes=0,
            reserved_bytes=0,
            peak_allocated_bytes=0,
            num_allocations=0
        )
        self._snapshots.append(snapshot)
    
    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all memory snapshots."""
        return list(self._snapshots)
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage."""
        if not self._snapshots:
            return 0
        return max(s.allocated_bytes for s in self._snapshots)
    
    def get_timeline(self) -> List[tuple]:
        """Get memory timeline as (timestamp, bytes) pairs."""
        return [(s.timestamp, s.allocated_bytes) for s in self._snapshots]


class LatencyProfiler:
    """
    Latency profiler for measuring operation latencies.
    
    Example:
        >>> latency_profiler = LatencyProfiler()
        >>> 
        >>> for batch in batches:
        ...     with latency_profiler.measure("decode"):
        ...         output = model.decode(batch)
        >>> 
        >>> stats = latency_profiler.get_stats()
        >>> print(f"P99 latency: {stats['decode']['p99']:.2f} ms")
    """
    
    def __init__(self):
        self._measurements: Dict[str, List[float]] = defaultdict(list)
    
    @contextlib.contextmanager
    def measure(self, name: str):
        """Measure latency of an operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._measurements[name].append(elapsed)
    
    def record(self, name: str, latency_ms: float):
        """Record a latency measurement."""
        self._measurements[name].append(latency_ms)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics."""
        stats = {}
        
        for name, latencies in self._measurements.items():
            if not latencies:
                continue
            
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            stats[name] = {
                'count': n,
                'mean': sum(latencies) / n,
                'min': sorted_latencies[0],
                'max': sorted_latencies[-1],
                'p50': sorted_latencies[int(n * 0.50)],
                'p90': sorted_latencies[int(n * 0.90)],
                'p95': sorted_latencies[int(n * 0.95)],
                'p99': sorted_latencies[min(int(n * 0.99), n - 1)],
            }
        
        return stats
    
    def reset(self):
        """Reset all measurements."""
        self._measurements.clear()


class ThroughputMonitor:
    """
    Monitor for tracking throughput metrics.
    
    Example:
        >>> monitor = ThroughputMonitor()
        >>> monitor.start()
        >>> 
        >>> for batch in batches:
        ...     outputs = model.generate(batch)
        ...     monitor.record_tokens(len(outputs))
        >>> 
        >>> monitor.stop()
        >>> print(f"Throughput: {monitor.get_throughput():.1f} tokens/s")
    """
    
    def __init__(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._total_tokens = 0
        self._total_requests = 0
        self._token_times: List[tuple] = []
    
    def start(self):
        """Start monitoring."""
        self._start_time = time.perf_counter()
        self._total_tokens = 0
        self._total_requests = 0
        self._token_times.clear()
    
    def stop(self):
        """Stop monitoring."""
        self._end_time = time.perf_counter()
    
    def record_tokens(self, num_tokens: int):
        """Record generated tokens."""
        self._total_tokens += num_tokens
        self._token_times.append((time.perf_counter(), num_tokens))
    
    def record_request(self, num_tokens: int):
        """Record a completed request."""
        self._total_requests += 1
        self.record_tokens(num_tokens)
    
    def get_throughput(self) -> float:
        """Get overall throughput in tokens/second."""
        elapsed = self._end_time - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_tokens / elapsed
    
    def get_request_rate(self) -> float:
        """Get request rate in requests/second."""
        elapsed = self._end_time - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_requests / elapsed
    
    def get_stats(self) -> Dict[str, float]:
        """Get all throughput stats."""
        elapsed = self._end_time - self._start_time
        return {
            'total_tokens': self._total_tokens,
            'total_requests': self._total_requests,
            'elapsed_seconds': elapsed,
            'tokens_per_second': self.get_throughput(),
            'requests_per_second': self.get_request_rate(),
        }


#===----------------------------------------------------------------------===#
# Decorators
#===----------------------------------------------------------------------===#

def profile(name: Optional[str] = None, 
            event_type: EventType = EventType.CUSTOM):
    """
    Decorator for profiling functions.
    
    Example:
        >>> @profile("my_function", EventType.CUSTOM)
        ... def my_function(x):
        ...     return x * 2
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            profiler = Profiler.get_instance()
            with profiler.trace(func_name, event_type):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def trace(name: str, event_type: EventType = EventType.CUSTOM):
    """
    Context manager for tracing code blocks.
    
    Example:
        >>> with trace("attention_block", EventType.ATTENTION):
        ...     output = attention(q, k, v)
    """
    profiler = Profiler.get_instance()
    return profiler.trace(name, event_type)
