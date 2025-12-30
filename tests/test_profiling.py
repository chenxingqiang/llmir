"""Tests for LLMIR profiling module."""

import time
import pytest

from llmir.profiling import (
    Profiler,
    MemoryProfiler,
    LatencyProfiler,
    ThroughputMonitor,
    ProfileReport,
    EventType,
    profile,
    trace,
)


class TestProfiler:
    """Tests for Profiler."""
    
    def test_create_profiler(self):
        """Test profiler creation."""
        profiler = Profiler()
        assert not profiler.is_active()
    
    def test_start_stop(self):
        """Test start and stop."""
        profiler = Profiler()
        
        profiler.start()
        assert profiler.is_active()
        
        profiler.stop()
        assert not profiler.is_active()
    
    def test_trace_context(self):
        """Test trace context manager."""
        profiler = Profiler()
        profiler.start()
        
        with profiler.trace("test_op", EventType.ATTENTION):
            time.sleep(0.01)  # 10ms
        
        profiler.stop()
        report = profiler.get_report()
        
        assert len(report.events) == 1
        assert report.events[0].name == "test_op"
        assert report.events[0].duration_ms >= 10
    
    def test_multiple_traces(self):
        """Test multiple traces."""
        profiler = Profiler()
        profiler.start()
        
        for i in range(5):
            with profiler.trace("operation"):
                time.sleep(0.001)
        
        profiler.stop()
        report = profiler.get_report()
        
        assert len(report.events) == 5
    
    def test_record_event(self):
        """Test recording events directly."""
        profiler = Profiler()
        profiler.start()
        
        profiler.record_event("custom_event", EventType.CUSTOM, 5.0)
        
        profiler.stop()
        report = profiler.get_report()
        
        assert len(report.events) == 1
        assert report.events[0].duration_ms == 5.0
    
    def test_singleton(self):
        """Test singleton instance."""
        p1 = Profiler.get_instance()
        p2 = Profiler.get_instance()
        
        assert p1 is p2
    
    def test_reset(self):
        """Test reset clears events."""
        profiler = Profiler()
        profiler.start()
        
        with profiler.trace("op1"):
            pass
        
        profiler.reset()
        profiler.stop()
        
        report = profiler.get_report()
        assert len(report.events) == 0
    
    def test_inactive_trace_noop(self):
        """Test trace is noop when inactive."""
        profiler = Profiler()
        
        with profiler.trace("test"):
            pass
        
        report = profiler.get_report()
        assert len(report.events) == 0


class TestProfileReport:
    """Tests for ProfileReport."""
    
    def test_get_stats(self):
        """Test statistics calculation."""
        profiler = Profiler()
        profiler.start()
        
        for _ in range(10):
            with profiler.trace("op"):
                time.sleep(0.001)
        
        profiler.stop()
        report = profiler.get_report()
        stats = report.get_stats()
        
        assert "op" in stats
        assert stats["op"].count == 10
        assert stats["op"].total_ms > 0
        assert stats["op"].avg_ms > 0
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        profiler = Profiler()
        profiler.start()
        
        with profiler.trace("test"):
            pass
        
        profiler.stop()
        report = profiler.get_report()
        d = report.to_dict()
        
        assert "total_time_ms" in d
        assert "num_events" in d
        assert "stats" in d


class TestMemoryProfiler:
    """Tests for MemoryProfiler."""
    
    def test_create_memory_profiler(self):
        """Test memory profiler creation."""
        profiler = MemoryProfiler()
        assert profiler.sample_interval > 0
    
    def test_record_snapshot(self):
        """Test recording memory snapshots."""
        profiler = MemoryProfiler()
        profiler.start()
        
        profiler.record(allocated=1024 * 1024, reserved=2 * 1024 * 1024)
        profiler.record(allocated=2 * 1024 * 1024, reserved=3 * 1024 * 1024)
        
        profiler.stop()
        
        peak = profiler.get_peak_memory()
        assert peak == 2 * 1024 * 1024
    
    def test_timeline(self):
        """Test getting memory timeline."""
        profiler = MemoryProfiler()
        profiler.start()
        
        profiler.record(1000)
        profiler.record(2000)
        profiler.record(1500)
        
        profiler.stop()
        
        timeline = profiler.get_timeline()
        assert len(timeline) == 3


class TestLatencyProfiler:
    """Tests for LatencyProfiler."""
    
    def test_measure_context(self):
        """Test measure context manager."""
        profiler = LatencyProfiler()
        
        with profiler.measure("operation"):
            time.sleep(0.01)
        
        stats = profiler.get_stats()
        
        assert "operation" in stats
        assert stats["operation"]["count"] == 1
        assert stats["operation"]["mean"] >= 10
    
    def test_record_latency(self):
        """Test recording latency directly."""
        profiler = LatencyProfiler()
        
        profiler.record("op", 5.0)
        profiler.record("op", 10.0)
        profiler.record("op", 15.0)
        
        stats = profiler.get_stats()
        
        assert stats["op"]["count"] == 3
        assert stats["op"]["mean"] == 10.0
        assert stats["op"]["min"] == 5.0
        assert stats["op"]["max"] == 15.0
    
    def test_percentiles(self):
        """Test percentile calculation."""
        profiler = LatencyProfiler()
        
        for i in range(100):
            profiler.record("op", float(i))
        
        stats = profiler.get_stats()
        
        assert stats["op"]["p50"] == 50.0
        assert stats["op"]["p90"] == 90.0
        assert stats["op"]["p95"] == 95.0
    
    def test_reset(self):
        """Test reset clears measurements."""
        profiler = LatencyProfiler()
        
        profiler.record("op", 5.0)
        profiler.reset()
        
        stats = profiler.get_stats()
        assert "op" not in stats


class TestThroughputMonitor:
    """Tests for ThroughputMonitor."""
    
    def test_record_tokens(self):
        """Test recording tokens."""
        monitor = ThroughputMonitor()
        monitor.start()
        
        monitor.record_tokens(100)
        monitor.record_tokens(200)
        
        time.sleep(0.01)
        monitor.stop()
        
        stats = monitor.get_stats()
        assert stats["total_tokens"] == 300
    
    def test_record_request(self):
        """Test recording requests."""
        monitor = ThroughputMonitor()
        monitor.start()
        
        monitor.record_request(50)
        monitor.record_request(50)
        
        time.sleep(0.01)
        monitor.stop()
        
        stats = monitor.get_stats()
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 100
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        monitor = ThroughputMonitor()
        monitor.start()
        
        monitor.record_tokens(1000)
        
        time.sleep(0.1)  # 100ms
        monitor.stop()
        
        throughput = monitor.get_throughput()
        # Should be around 10000 tokens/sec
        assert throughput > 5000


class TestDecorators:
    """Tests for profiling decorators."""
    
    def test_profile_decorator(self):
        """Test profile decorator."""
        @profile("decorated_func", EventType.CUSTOM)
        def my_function():
            time.sleep(0.001)
            return 42
        
        profiler = Profiler.get_instance()
        profiler.start()
        
        result = my_function()
        
        profiler.stop()
        
        assert result == 42
        report = profiler.get_report()
        assert any(e.name == "decorated_func" for e in report.events)
    
    def test_trace_function(self):
        """Test trace context function."""
        profiler = Profiler.get_instance()
        profiler.start()
        
        with trace("traced_block"):
            time.sleep(0.001)
        
        profiler.stop()
        
        report = profiler.get_report()
        assert any(e.name == "traced_block" for e in report.events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
