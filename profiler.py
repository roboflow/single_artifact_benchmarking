import torch
from typing import List, Optional, Dict
from contextlib import contextmanager
import numpy as np


class CUDAProfiler:
    """Hardware-accurate CUDA profiler with context manager support"""
    
    def __init__(self):
        self.timings: List[float] = []
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        
    @contextmanager
    def profile(self):
        """Context manager for profiling GPU operations"""
        # Record start event
        torch.cuda.synchronize()
        self._start_event.record()
        
        # Yield control back to the caller
        yield
        
        # Record end event
        self._end_event.record()
        torch.cuda.synchronize()
        
        # Calculate and store elapsed time
        elapsed_ms = self._start_event.elapsed_time(self._end_event)
        self.timings.append(elapsed_ms)
    
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
    
    def reset(self):
        """Clear all recorded timings"""
        self.timings.clear()
    
    def get_last_timing(self) -> float:
        """Get the most recent timing"""
        if not self.timings:
            raise ValueError("No timings recorded.")
        return self.timings[-1]
