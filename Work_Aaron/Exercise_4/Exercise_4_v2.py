import cupy as cp
import numpy as np
import time

def benchmark_with_transfers():
    """Matrix multiplication with explicit H2D and D2H transfers using streams"""
    n_iterations = 20
    size = 4096
    
    # Create three streams: one for H2D, one for compute, one for D2H
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    compute_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)
    
    print("Running stream-based multiplication with explicit transfers...")
    
    total_times = []
    transfer_times = []
    compute_times = []
    
    for i in range(n_iterations):
        start_total = time.time()
        
        # Create host matrices
        host_a = np.random.rand(size, size).astype(np.float32)
        host_b = np.random.rand(size, size).astype(np.float32)
        host_c = np.empty((size, size), dtype=np.float32)
        
        # Time H2D transfer
        h2d_start = time.time()
        with h2d_stream:
            # Host to Device transfer
            dev_a = cp.asarray(host_a)
            dev_b = cp.asarray(host_b)
        h2d_stream.synchronize()
        h2d_time = time.time() - h2d_start
        
        # Time computation
        compute_start = time.time()
        with compute_stream:
            # Wait for H2D to complete
            h2d_stream.synchronize()
            # Perform multiplication
            dev_c = cp.matmul(dev_a, dev_b)
        compute_stream.synchronize()
        compute_time = time.time() - compute_start
        
        # Time D2H transfer
        d2h_start = time.time()
        with d2h_stream:
            # Wait for computation to complete
            compute_stream.synchronize()
            # Device to Host transfer
            dev_c.get(out=host_c)
        d2h_stream.synchronize()
        d2h_time = time.time() - d2h_start
        
        total_time = time.time() - start_total
        
        # Record times
        transfer_times.append(h2d_time + d2h_time)
        compute_times.append(compute_time)
        total_times.append(total_time)
        
        print(f"\nIteration {i+1}:")
        print(f"H2D transfer time: {h2d_time:.3f} seconds")
        print(f"Compute time: {compute_time:.3f} seconds")
        print(f"D2H transfer time: {d2h_time:.3f} seconds")
        print(f"Total time: {total_time:.3f} seconds")
    
    print(f"\nAverage times over {n_iterations} iterations:")
    print(f"Transfer time (H2D + D2H): {np.mean(transfer_times):.3f} seconds")
    print(f"Compute time: {np.mean(compute_times):.3f} seconds")
    print(f"Total time: {np.mean(total_times):.3f} seconds")

def benchmark_with_overlapping():
    """Matrix multiplication with overlapping H2D, compute, and D2H operations"""
    n_iterations = 20
    size = 4096
    
    # Create streams
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    compute_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)
    
    print("\nRunning overlapped stream-based multiplication...")
    
    # Pre-allocate host and device arrays
    host_matrices_a = [np.random.rand(size, size).astype(np.float32) for _ in range(n_iterations)]
    host_matrices_b = [np.random.rand(size, size).astype(np.float32) for _ in range(n_iterations)]
    host_results = [np.empty((size, size), dtype=np.float32) for _ in range(n_iterations)]
    
    dev_a = cp.empty((size, size), dtype=cp.float32)
    dev_b = cp.empty((size, size), dtype=cp.float32)
    dev_c = cp.empty((size, size), dtype=cp.float32)
    
    start_total = time.time()
    
    for i in range(n_iterations):
        # H2D transfer for next iteration
        with h2d_stream:
            cp.copyto(dev_a, cp.asarray(host_matrices_a[i]))
            cp.copyto(dev_b, cp.asarray(host_matrices_b[i]))
        
        # Computation
        with compute_stream:
            h2d_stream.synchronize()  # Wait for H2D to complete
            dev_c = cp.matmul(dev_a, dev_b)
        
        # D2H transfer of previous result
        with d2h_stream:
            compute_stream.synchronize()  # Wait for computation to complete
            dev_c.get(out=host_results[i])
    
    # Wait for all operations to complete
    d2h_stream.synchronize()
    
    total_time = time.time() - start_total
    print(f"Total time with overlapping: {total_time:.3f} seconds")
    print(f"Average time per iteration: {total_time/n_iterations:.3f} seconds")

def main():
    # Warm up the GPU
    warmup = cp.random.rand(100, 100, dtype=cp.float32)
    cp.matmul(warmup, warmup)
    cp.cuda.Stream.null.synchronize()
    
    print("Starting benchmarks...\n")
    
    # Run non-overlapping benchmark
    benchmark_with_transfers()
    
    # Run overlapping benchmark
    benchmark_with_overlapping()
    
    # Print memory usage
    print("\nMemory usage statistics:")
    print(f"Used memory: {cp.get_default_memory_pool().used_bytes() / 1024**2:.2f} MB")
    print(f"Total memory: {cp.get_default_memory_pool().total_bytes() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()