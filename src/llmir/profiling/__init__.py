"""
LLMIR Profiling Module

Performance profiling tools for analyzing LLM inference.
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
    'MemoryProfiler',
    'LatencyProfiler',
    'ThroughputMonitor',
    'ProfileReport',
    'EventType',
    'profile',
    'trace',
]


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
class ProfileEvent:
    """A single profiling event."""
    name: str
    event_type: EventType
    start_time: float
    end_time: float
    duration_ms: float
    thread_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    def __init__(self):
        self._events: List[ProfileEvent] = []
        self._active = False
        self._start_time = 0.0
        self._end_time = 0.0
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
        """Context manager for tracing an operation."""
        if not self._active:
            yield
            return
        
        thread_id = threading.get_ident()
        start_time = time.perf_counter()
        
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
    
    def record_event(self, name: str, event_type: EventType,
                     duration_ms: float, **metadata):
        """Record a single event."""
        if not self._active:
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
    
    def get_report(self) -> 'ProfileReport':
        """Generate a profile report."""
        return ProfileReport(
            events=self._events,
            counters=dict(self._counters),
            total_time_ms=(self._end_time - self._start_time) * 1000
        )
    
    def export_chrome_trace(self, filepath: str):
        """Export trace in Chrome trace format."""
        chrome_events = []
        for e in self._events:
            chrome_events.append({
                'name': e.name,
                'cat': e.event_type.name,
                'ph': 'X',
                'ts': int(e.start_time * 1e6),
                'dur': int(e.duration_ms * 1e3),
                'tid': e.thread_id,
                'pid': os.getpid(),
                'args': e.metadata,
            })
        
        with open(filepath, 'w') as f:
            json.dump({'traceEvents': chrome_events}, f)
    
    def reset(self):
        """Reset all recorded data."""
        with self._lock:
            self._events.clear()
            self._counters.clear()


class ProfileReport:
    """Profile report with analysis."""
    
    def __init__(self, events: List[ProfileEvent], 
                 counters: Dict[str, int], total_time_ms: float):
        self.events = events
        self.counters = counters
        self.total_time_ms = total_time_ms
        self._stats: Optional[Dict[str, EventStats]] = None
    
    def get_stats(self) -> Dict[str, EventStats]:
        """Compute statistics for each event type."""
        if self._stats is not None:
            return self._stats
        
        events_by_name: Dict[str, List[float]] = defaultdict(list)
        for event in self.events:
            events_by_name[event.name].append(event.duration_ms)
        
        self._stats = {}
        for name, durations in events_by_name.items():
            n = len(durations)
            total = sum(durations)
            avg = total / n if n > 0 else 0
            variance = sum((d - avg) ** 2 for d in durations) / n if n > 1 else 0
            
            self._stats[name] = EventStats(
                name=name,
                count=n,
                total_ms=total,
                avg_ms=avg,
                min_ms=min(durations) if durations else 0,
                max_ms=max(durations) if durations else 0,
                std_ms=variance ** 0.5,
                percentage=(total / self.total_time_ms * 100) if self.total_time_ms > 0 else 0
            )
        
        return self._stats
    
    def print_summary(self):
        """Print a summary of the profile."""
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print("LLMIR Performance Profile Summary")
        print("=" * 70)
        print(f"\nTotal Time: {self.total_time_ms:.2f} ms")
        print(f"Total Events: {len(self.events)}")
        print()
        
        sorted_stats = sorted(stats.values(), key=lambda x: -x.total_ms)
        
        header = f"{'Operation':<25} {'Count':>8} {'Total':>10} {'Avg':>8} {'%':>6}"
        print(header)
        print("-" * 60)
        
        for stat in sorted_stats:
            print(f"{stat.name:<25} {stat.count:>8} {stat.total_ms:>9.2f}ms "
                  f"{stat.avg_ms:>7.2f}ms {stat.percentage:>5.1f}%")
    
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
            } for name, s in stats.items()},
        }


class MemoryProfiler:
    """Memory profiler for tracking allocations."""
    
    def __init__(self, sample_interval_ms: float = 100):
        self.sample_interval = sample_interval_ms / 1000
        self._snapshots: List[Dict[str, int]] = []
        self._active = False
    
    def start(self):
        """Start memory profiling."""
        self._active = True
        self._snapshots.clear()
    
    def stop(self):
        """Stop memory profiling."""
        self._active = False
    
    def record(self, allocated: int, reserved: int = 0):
        """Record a memory snapshot."""
        if self._active:
            self._snapshots.append({
                'timestamp': time.perf_counter(),
                'allocated': allocated,
                'reserved': reserved,
            })
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage."""
        if not self._snapshots:
            return 0
        return max(s['allocated'] for s in self._snapshots)
    
    def get_timeline(self) -> List[tuple]:
        """Get memory timeline."""
        return [(s['timestamp'], s['allocated']) for s in self._snapshots]


class LatencyProfiler:
    """Latency profiler with percentile statistics."""
    
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
        """Get latency statistics with percentiles."""
        stats = {}
        for name, latencies in self._measurements.items():
            if not latencies:
                continue
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            stats[name] = {
                'count': n,
                'mean': sum(latencies) / n,
                'min': sorted_lat[0],
                'max': sorted_lat[-1],
                'p50': sorted_lat[int(n * 0.50)],
                'p90': sorted_lat[int(n * 0.90)],
                'p95': sorted_lat[int(n * 0.95)],
                'p99': sorted_lat[min(int(n * 0.99), n - 1)],
            }
        return stats
    
    def reset(self):
        """Reset all measurements."""
        self._measurements.clear()


class ThroughputMonitor:
    """Monitor for tracking throughput metrics."""
    
    def __init__(self):
        self._start_time = 0.0
        self._end_time = 0.0
        self._total_tokens = 0
        self._total_requests = 0
    
    def start(self):
        """Start monitoring."""
        self._start_time = time.perf_counter()
        self._total_tokens = 0
        self._total_requests = 0
    
    def stop(self):
        """Stop monitoring."""
        self._end_time = time.perf_counter()
    
    def record_tokens(self, num_tokens: int):
        """Record generated tokens."""
        self._total_tokens += num_tokens
    
    def record_request(self, num_tokens: int = 0):
        """Record a completed request."""
        self._total_requests += 1
        if num_tokens:
            self._total_tokens += num_tokens
    
    def get_throughput(self) -> float:
        """Get throughput in tokens/second."""
        elapsed = self._end_time - self._start_time
        return self._total_tokens / elapsed if elapsed > 0 else 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get all throughput stats."""
        elapsed = self._end_time - self._start_time
        return {
            'total_tokens': self._total_tokens,
            'total_requests': self._total_requests,
            'elapsed_seconds': elapsed,
            'tokens_per_second': self.get_throughput(),
            'requests_per_second': self._total_requests / elapsed if elapsed > 0 else 0,
        }


def profile(name: Optional[str] = None, 
            event_type: EventType = EventType.CUSTOM):
    """Decorator for profiling functions."""
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        def wrapper(*args, **kwargs):
            profiler = Profiler.get_instance()
            with profiler.trace(func_name, event_type):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def trace(name: str, event_type: EventType = EventType.CUSTOM):
    """Context manager for tracing code blocks."""
    profiler = Profiler.get_instance()
    return profiler.trace(name, event_type)
