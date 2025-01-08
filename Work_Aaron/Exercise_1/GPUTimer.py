import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Any
import cupy as cp
from numba import cuda

class GPUTimer:
    def __init__(self, backend='cupy'):
        """
        Initialize GPU Timer with specified backend ('cupy' or 'numba')
        """
        self.backend = backend.lower()
        
        # Check CUDA availability based on backend
        if self.backend == 'cupy':
            try:
                self.num_gpus = cp.cuda.runtime.getDeviceCount()
                self.cuda_available = True
                print(f"CuPy detected {self.num_gpus} CUDA device(s)")
                # Print device information
                for i in range(self.num_gpus):
                    device = cp.cuda.runtime.getDeviceProperties(i)
                    print(f"Device {i}: {device['name'].decode()}")
            except:
                self.cuda_available = False
                print("CUDA not available for CuPy")
                
        elif self.backend == 'numba':
            self.cuda_available = cuda.is_available()
            if self.cuda_available:
                print("CUDA available for Numba")
                print(f"Current device: {cuda.get_current_device().name}")
            else:
                print("CUDA not available for Numba")
        else:
            raise ValueError("Backend must be either 'cupy' or 'numba'")
            
        if not self.cuda_available:
            raise RuntimeError("CUDA is not available for the selected backend")
            
        # Initialize storage for measurements
        self.measurements: List[float] = []
    
    def _warmup_gpu(self, func: Callable[[], Any], num_warmup: int = 10):
        """Warm up the GPU using the target function"""
        print("Warming up GPU...")
        for i in range(num_warmup):
            func()  # Run the actual function for warmup
            if self.backend == 'cupy':
                cp.cuda.stream.get_current_stream().synchronize()
            else:  # numba
                cuda.synchronize()
        print("Warmup complete")
    
    def measure_execution(self, func: Callable[[], Any], num_runs: int = 3) -> List[float]:
        """Measure execution time of a function multiple times"""
        # First do warmup with the provided function
        self._warmup_gpu(func)
        
        self.measurements = []
        
        for i in range(num_runs):
            # Clear GPU cache if using CuPy
            if self.backend == 'cupy':
                cp.get_default_memory_pool().free_all_blocks()
            
            # Ensure all GPU operations are completed
            if self.backend == 'cupy':
                cp.cuda.stream.get_current_stream().synchronize()
            else:  # numba
                cuda.synchronize()
            
            # Measure execution time
            start = time.perf_counter()
            
            func()  # Execute the function
            
            # Ensure all GPU operations are completed before stopping timer
            if self.backend == 'cupy':
                cp.cuda.stream.get_current_stream().synchronize()
            else:  # numba
                cuda.synchronize()
                
            end = time.perf_counter()
            
            execution_time = (end - start) * 1000  # Convert to milliseconds
            self.measurements.append(execution_time)
            print(f"Run {i+1}: {execution_time:.2f} ms")
        
        return self.measurements
    
    def visualize_results(self):
        """Visualize the measurement results"""
        if not self.measurements:
            print("No measurements available to visualize")
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(self.measurements) + 1), self.measurements)
        plt.axhline(y=np.mean(self.measurements), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(self.measurements):.2f} ms')
        
        plt.xlabel('Run Number')
        plt.ylabel('Execution Time (ms)')
        plt.title(f'GPU Function Execution Time Measurements ({self.backend.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print statistics
        print("\nStatistics:")
        print(f"Mean execution time: {np.mean(self.measurements):.2f} ms")
        print(f"Std deviation: {np.std(self.measurements):.2f} ms")
        print(f"Min execution time: {np.min(self.measurements):.2f} ms")
        print(f"Max execution time: {np.max(self.measurements):.2f} ms")