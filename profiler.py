import torch
from typing import List, Optional, Dict
from contextlib import contextmanager
import numpy as np


class CUDAProfiler:
    """Hardware-accurate CUDA profiler with stream-aware timing
    
    This profiler provides two timing modes:
    1. Synchronous profiling for standard operations
    2. Asynchronous profiling for CUDA graphs (avoids interference)
    """
    
    def __init__(self, stream: Optional[torch.cuda.Stream] = None):
        self.timings: List[float] = []
        self.stream = stream  # Allow profiling on specific streams
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        
    def set_stream(self, stream: torch.cuda.Stream):
        """Set the stream to profile on"""
        self.stream = stream
        
    @contextmanager
    def profile(self, stream: Optional[torch.cuda.Stream] = None):
        """Synchronous profiling for standard operations
        
        Use this for regular PyTorch operations, TensorRT standard execution, etc.
        This method handles synchronization automatically.
        
        Args:
            stream: Optional stream to profile on. If None, uses self.stream or current stream.
        """
        # Use provided stream, or fall back to instance stream, or current stream
        target_stream = stream or self.stream
        
        if target_stream is not None:
            # Record events on the specific stream for accurate timing
            self._start_event.record(target_stream)
            
            # Yield control back to the caller
            yield
            
            # Record end event on the same stream
            self._end_event.record(target_stream)
            
            # Only synchronize the target stream (more efficient than global sync)
            target_stream.synchronize()
        else:
            # Fallback to default stream timing
            self._start_event.record()
            
            yield
            
            self._end_event.record()
            torch.cuda.synchronize()
        
        # Calculate and store elapsed time
        try:
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            self.timings.append(elapsed_ms)
        except RuntimeError as e:
            # Handle CUDA errors gracefully (e.g., context corruption)
            print(f"Timing measurement failed: {e}")
            # Don't append invalid timing
    
    @contextmanager 
    def profile_async(self, stream: Optional[torch.cuda.Stream] = None):
        """Asynchronous profiling for CUDA graphs and async operations
        
        Use this for CUDA graphs, async kernel launches, etc.
        This method records timing events but doesn't synchronize within the context.
        Call get_last_timing_async() after ensuring the stream is complete.
        
        Example:
            with profiler.profile_async(stream=my_stream):
                cuda_graph.replay()
            my_stream.synchronize()  # Ensure completion
            timing = profiler.get_last_timing_async()
        """
        target_stream = stream or self.stream
        
        if target_stream is not None:
            self._start_event.record(target_stream)
            
            yield
            
            self._end_event.record(target_stream)
            # Don't synchronize here - let caller handle it for async operations
        else:
            self._start_event.record()
            
            yield
            
            self._end_event.record()
    
    def get_last_timing_async(self) -> Optional[float]:
        """Get timing from async profiling after stream completion
        
        Call this after ensuring the stream has completed (e.g., via stream.synchronize()).
        
        Returns:
            Timing in milliseconds if available, None if events aren't ready
        """
        try:
            # Check if events are ready (non-blocking)
            if self._end_event.query():
                elapsed_ms = self._start_event.elapsed_time(self._end_event)
                self.timings.append(elapsed_ms)
                return elapsed_ms
            else:
                return None  # Events not ready yet
        except RuntimeError as e:
            print(f"Async timing measurement failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, float]:
        """Compute statistics from collected timings"""
        if not self.timings:
            raise ValueError("No timings recorded. Use .profile() context manager first.")
        
        timings_array = np.array(self.timings)
        
        return {
            'mean': np.mean(timings_array),
            'median': np.median(timings_array),
            'min': np.min(timings_array),
            'max': np.max(timings_array),
            'std': np.std(timings_array),
            'p90': np.percentile(timings_array, 90),
            'p95': np.percentile(timings_array, 95),
            'p99': np.percentile(timings_array, 99),
            'count': len(self.timings)
        }
    
    def print_stats(self, name: Optional[str] = None):
        """Pretty print statistics"""
        try:
            stats = self.get_stats()
            
            if name:
                print(f"\n=== {name} ===")
            else:
                print("\n=== Profiling Results ===")
                
            print(f"Samples: {stats['count']}")
            print(f"Mean:    {stats['mean']:.3f} ms")
            print(f"Median:  {stats['median']:.3f} ms")
            print(f"Min:     {stats['min']:.3f} ms")
            print(f"Max:     {stats['max']:.3f} ms")
            print(f"Std:     {stats['std']:.3f} ms")
            print(f"P90:     {stats['p90']:.3f} ms")
            print(f"P95:     {stats['p95']:.3f} ms")
            print(f"P99:     {stats['p99']:.3f} ms")
        except ValueError as e:
            print(f"Cannot print stats: {e}")
    
    def reset(self):
        """Clear all recorded timings"""
        self.timings.clear()
    
    def get_last_timing(self) -> float:
        """Get the most recent timing"""
        if not self.timings:
            raise ValueError("No timings recorded.")
        return self.timings[-1]


# Additional utility for benchmarking specific operations
class StreamAwareBenchmark:
    """Utility for benchmarking operations with proper stream handling"""
    
    @staticmethod
    def time_operation(operation_fn, stream: torch.cuda.Stream, iterations: int = 100, use_async: bool = False):
        """Time a specific operation on a stream
        
        Args:
            operation_fn: Function to time
            stream: CUDA stream to run on
            iterations: Number of iterations 
            use_async: Whether to use async profiling (for CUDA graphs)
        """
        profiler = CUDAProfiler(stream)
        
        # Warmup
        for _ in range(10):
            with torch.cuda.stream(stream):
                operation_fn()
            stream.synchronize()
        
        # Actual timing
        for _ in range(iterations):
            with torch.cuda.stream(stream):
                if use_async:
                    with profiler.profile_async():
                        operation_fn()
                    stream.synchronize()
                    profiler.get_last_timing_async()
                else:
                    with profiler.profile():
                        operation_fn()
        
        return profiler.get_stats()
    
    @staticmethod
    def compare_cuda_graph_vs_standard(cuda_graph_fn, standard_fn, stream: torch.cuda.Stream, iterations: int = 100):
        """Compare CUDA graph vs standard execution performance"""
        
        print("Benchmarking Standard Execution...")
        stats_standard = StreamAwareBenchmark.time_operation(standard_fn, stream, iterations, use_async=False)
        
        print("Benchmarking CUDA Graph Execution...")
        stats_graph = StreamAwareBenchmark.time_operation(cuda_graph_fn, stream, iterations, use_async=True)
        
        print(f"\nStandard Execution - Mean: {stats_standard['mean']:.3f} ms")
        print(f"CUDA Graph Execution - Mean: {stats_graph['mean']:.3f} ms")
        print(f"CUDA Graph Speedup: {stats_standard['mean'] / stats_graph['mean']:.2f}x")
        
        return stats_standard, stats_graph