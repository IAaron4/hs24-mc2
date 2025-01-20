"""
Matrix multiplication benchmark comparing CPU, basic GPU, and shared memory GPU implementations.
This script provides a comprehensive comparison of different matrix multiplication approaches:
1. CPU implementation using numpy
2. Basic GPU implementation without shared memory
3. Optimized GPU implementation with shared memory
4. GPU matrix multiplication using CuPy

The script measures execution time and accuracy for different matrix and block sizes.
"""

import numpy as np
import pandas as pd
from numba import cuda, float32
import math
import time
import cupy as cp
from typing import Dict, List, Tuple

def cpu_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    CPU implementation of matrix multiplication using numpy.
    This serves as our baseline for performance comparison.
    
    Args:
        A (ndarray): First input matrix of shape (M, K)
        B (ndarray): Second input matrix of shape (K, N)
    
    Returns:
        ndarray: Result of matrix multiplication A @ B of shape (M, N)
    """
    return np.matmul(A, B)

@cuda.jit
def basic_matmul(A, B, C):
    """
    Basic CUDA kernel for matrix multiplication without shared memory.
    This implementation directly accesses global memory for all operations.
    
    Memory access pattern:
    - Each thread reads multiple elements from A and B from global memory
    - Each thread writes one element to C in global memory
    
    Args:
        A (device array): Input matrix of shape (M, K)
        B (device array): Input matrix of shape (K, N)
        C (device array): Output matrix of shape (M, N)
    """
    # Calculate the row and column for this thread
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # Check if this thread should compute an element
    if row < C.shape[0] and col < C.shape[1]:
        # Initialize accumulator for dot product
        tmp = 0.0
        # Compute dot product of row from A and column from B
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        # Store the result
        C[row, col] = tmp

@cuda.jit
def shared_matmul(A, B, C, BLOCK_SIZE):
    """
    Optimized CUDA kernel for matrix multiplication using shared memory.
    This implementation uses a tiled approach to leverage fast shared memory.
    
    Memory access pattern:
    1. Cooperatively load a tile of A and B into shared memory
    2. Each thread computes partial results using shared memory
    3. Load next tile and repeat until done
    
    Args:
        A (device array): Input matrix of shape (M, K)
        B (device array): Input matrix of shape (K, N)
        C (device array): Output matrix of shape (M, N)
        BLOCK_SIZE (int): Size of the thread block (both dimensions)
    """
    # Allocate shared memory for the A and B tiles
    # These provide fast access for the block's threads
    sA = cuda.shared.array(shape=(32, 32), dtype=float32)
    sB = cuda.shared.array(shape=(32, 32), dtype=float32)
    
    # Get thread and block indices
    tx = cuda.threadIdx.x     # Thread x-index within block
    ty = cuda.threadIdx.y     # Thread y-index within block
    bx = cuda.blockIdx.x      # Block x-index in grid
    by = cuda.blockIdx.y      # Block y-index in grid
    
    # Compute global row and column indices for this thread
    row = by * BLOCK_SIZE + ty    # Global row index
    col = bx * BLOCK_SIZE + tx    # Global column index
    
    # Initialize accumulator for dot product
    tmp = float32(0.0)
    
    # Calculate number of tiles needed
    num_tiles = (A.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Process each tile
    for tile_idx in range(num_tiles):
        # Load tile from matrix A into shared memory
        if row < A.shape[0] and (tile_idx * BLOCK_SIZE + tx) < A.shape[1]:
            sA[ty, tx] = A[row, tile_idx * BLOCK_SIZE + tx]
        else:
            # Pad with zeros if outside matrix bounds
            sA[ty, tx] = 0
            
        # Load tile from matrix B into shared memory
        if (tile_idx * BLOCK_SIZE + ty) < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[tile_idx * BLOCK_SIZE + ty, col]
        else:
            # Pad with zeros if outside matrix bounds
            sB[ty, tx] = 0
            
        # Wait for all threads to finish loading
        cuda.syncthreads()
        
        # Compute partial dot product for this tile
        for k in range(BLOCK_SIZE):
            tmp += sA[ty, k] * sB[k, tx]
            
        # Wait for all threads to finish computing
        cuda.syncthreads()
    
    # Store final result in global memory
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

def run_gpu_basic(A: np.ndarray, B: np.ndarray, block_size: int,
                 num_warmup: int = 3, num_runs: int = 5) -> Dict:
    """
    Run basic GPU matrix multiplication (without shared memory)
    
    Args:
        A (ndarray): First input matrix
        B (ndarray): Second input matrix
        block_size (int): Size of thread blocks
        num_warmup (int): Number of warmup runs
        num_runs (int): Number of timed runs
    
    Returns:
        Dict containing timing statistics and accuracy metrics
    """
    M, K = A.shape
    K, N = B.shape
    
    # Transfer matrices to GPU memory
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.device_array((M, N), dtype=np.float32)
    
    # Configure grid dimensions
    threadsperblock = (block_size, block_size)
    blockspergrid_x = math.ceil(M / block_size)
    blockspergrid_y = math.ceil(N / block_size)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Perform warmup runs
    for _ in range(num_warmup):
        basic_matmul[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu)
        cuda.synchronize()
    
    # Perform timed runs
    times = []
    for _ in range(num_runs):
        cuda.synchronize()
        start = time.perf_counter()
        basic_matmul[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu)
        cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    # Verify results against CPU implementation
    C_gpu_result = C_gpu.copy_to_host()
    C_cpu = np.matmul(A, B)
    max_diff = np.max(np.abs(C_gpu_result - C_cpu))
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'max_diff': max_diff
    }

def run_gpu_shared(A: np.ndarray, B: np.ndarray, block_size: int,
                  num_warmup: int = 3, num_runs: int = 5) -> Dict:
    """
    Run optimized GPU matrix multiplication (with shared memory)
    
    Args:
        A (ndarray): First input matrix
        B (ndarray): Second input matrix
        block_size (int): Size of thread blocks
        num_warmup (int): Number of warmup runs
        num_runs (int): Number of timed runs
    
    Returns:
        Dict containing timing statistics and accuracy metrics
    """
    M, K = A.shape
    K, N = B.shape
    
    # Transfer matrices to GPU memory
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.device_array((M, N), dtype=np.float32)
    
    # Configure grid dimensions
    threadsperblock = (block_size, block_size)
    blockspergrid_x = math.ceil(M / block_size)
    blockspergrid_y = math.ceil(N / block_size)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Perform warmup runs
    for _ in range(num_warmup):
        shared_matmul[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu, block_size)
        cuda.synchronize()
    
    # Perform timed runs
    times = []
    for _ in range(num_runs):
        cuda.synchronize()
        start = time.perf_counter()
        shared_matmul[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu, block_size)
        cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    # Verify results against CPU implementation
    C_gpu_result = C_gpu.copy_to_host()
    C_cpu = np.matmul(A, B)
    max_diff = np.max(np.abs(C_gpu_result - C_cpu))
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'max_diff': max_diff
    }

def run_gpu_optimized(A: np.ndarray, B: np.ndarray, block_size: int,
                 num_warmup: int = 3, num_runs: int = 5) -> Dict:
    """
    Run GPU matrix multiplication using CuPy
    
    Args:
        A (ndarray): First input matrix
        B (ndarray): Second input matrix
        block_size (int): Size of thread blocks (not used in CuPy implementation)
        num_warmup (int): Number of warmup runs
        num_runs (int): Number of timed runs
    
    Returns:
        Dict containing timing statistics and accuracy metrics
    """
    # Transfer matrices to GPU memory
    A_gpu = cp.asarray(A)
    B_gpu = cp.asarray(B)
    
    # Perform warmup runs
    for _ in range(num_warmup):
        C_gpu = cp.matmul(A_gpu, B_gpu)
        cp.cuda.stream.get_current_stream().synchronize()
    
    # Perform timed runs
    times = []
    for _ in range(num_runs):
        cp.cuda.stream.get_current_stream().synchronize()
        start = time.perf_counter()
        C_gpu = cp.matmul(A_gpu, B_gpu)
        cp.cuda.stream.get_current_stream().synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    # Verify results against CPU implementation
    C_gpu_result = cp.asnumpy(C_gpu)
    C_cpu = np.matmul(A, B)
    max_diff = np.max(np.abs(C_gpu_result - C_cpu))
    
    # Clear GPU memory
    del A_gpu, B_gpu, C_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'max_diff': max_diff
    }

def run_experiments(save_results: bool = True):
    """
    Run complete benchmark comparing CPU, basic GPU, and shared memory GPU implementations
    
    Args:
        save_results (bool): Whether to save results to CSV file
    
    Returns:
        DataFrame containing benchmark results for all implementations
    """
    results = []
    
    # Test each matrix size configuration
    for m1, m2, m3 in test_matrices:
        print(f"\nTesting matrix size: {m1}x{m2} * {m2}x{m3}")
        
        # Generate random input matrices
        A = np.random.randn(m1, m2).astype(np.float32)
        B = np.random.randn(m2, m3).astype(np.float32)

        # Measure CPU performance (baseline)
        print("\nRunning CPU baseline...")
        start = time.perf_counter()
        cpu_matmul(A, B)
        cpu_time = time.perf_counter() - start
        print(f"CPU Time: {cpu_time:.3f}s")

        # Test different block sizes
        test_experiment_on_block_size(A, B, BLOCK_SIZE, cpu_time, results, m1, m2, m3)
            
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if requested
    if save_results:
        results_df.to_csv('gpu_experiment_results.csv', index=False)
        
    return results_df

def test_experiment_on_block_size(A: np.ndarray, B: np.ndarray, block_size: int, cpu_time, results, m1, m2, m3):
    print(f"\nTesting block size: {block_size}x{block_size}")
    
    # Run basic GPU implementation (no shared memory)
    print("Running basic GPU implementation...")
    basic_results = run_gpu_basic(A, B, block_size)
    basic_speedup = cpu_time / basic_results['mean']
    
    # Run shared memory GPU implementation
    print("Running shared memory GPU implementation...")
    shared_results = run_gpu_shared(A, B, block_size)
    shared_speedup = cpu_time / shared_results['mean']
    
    # Run matrix multiplication with CuPy
    print("Running matrix multiplication with CuPy implementation...")
    cupy_results = run_gpu_optimized(A, B, block_size)
    cupy_speedup = cpu_time / cupy_results['mean']
    
    # Store results for this configuration
    results.append({
        'Matrix Size': f"{m1}x{m2}x{m3}",
        'Block Size': f"{block_size}x{block_size}",
        'CPU Time': cpu_time,
        'Basic GPU Time': basic_results['mean'],
        'Basic GPU Speedup': basic_speedup,
        'Basic GPU Max Diff': basic_results['max_diff'],
        'Shared GPU Time': shared_results['mean'],
        'Shared GPU Speedup': shared_speedup,
        'Shared GPU Max Diff': shared_results['max_diff'],
        'CuPy GPU Time': cupy_results['mean'],
        'CuPy GPU Speedup': cupy_speedup,
        'CuPy GPU Max Diff': cupy_results['max_diff'],
        'Elements': m1 * m2 * m3
    })
    
    # Print results for this configuration
    print(f"\nResults for {block_size}x{block_size} block size:")
    print(f"Basic GPU:   {basic_results['mean']:.4f}s (Speedup: {basic_speedup:.2f}x)")
    print(f"Shared GPU:  {shared_results['mean']:.4f}s (Speedup: {shared_speedup:.2f}x)")
    print(f"Cupy GPU:  {cupy_results['mean']:.4f}s (Speedup: {cupy_speedup:.2f}x)")


def visualize_results(df: pd.DataFrame):
    """
    Print detailed statistics about the experiment results
    
    Args:
        df (DataFrame): DataFrame containing benchmark results
    """
    print("\nDetailed Statistics:")
    print("=" * 80)
    
    # Group results by matrix size and block size
    stats = df.groupby(['Matrix Size', 'Block Size']).agg({
        'Basic GPU Time': ['mean', 'std'],
        'Basic GPU Speedup': 'mean',
        'Shared GPU Time': ['mean', 'std'],
        'Shared GPU Speedup': 'mean',
        'CuPy GPU Time': ['mean', 'std'],
        'CuPy GPU Speedup': 'mean',
        'Basic GPU Max Diff': 'max',
        'Shared GPU Max Diff': 'max',
        'CuPy GPU Max Diff': 'max',
    }).round(4)
    
    print("\nPerformance Statistics by Configuration:")
    print(stats)
    
    print("\nBest Configurations by Matrix Size:")
    
    # Find best configuration based on shared memory GPU speedup
    best_configs = df.loc[df.groupby('Matrix Size')['Shared GPU Speedup'].idxmax()]
    print(best_configs[['Matrix Size', 'Block Size', 'Basic GPU Speedup', 'Shared GPU Speedup']])

# Block sizes to test - can't use an array because of cuda shared error.
BLOCK_SIZE = 32

# Test matrix configurations for multiplication: (M,K) * (K,N) matrices
# Format: (M, K, N) where resulting matrix will be (M,N)
test_matrices = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (5120, 5120, 5120),     # ~26M elements - Medium size
    #(8192, 4096, 8192),     # ~34M elements - Medium-large size
    #(10240, 10240, 10240),  # ~105M elements - Large size
    #(16384, 8192, 8192),    # ~134M elements - Very large size
    #(20480, 20480, 20480)   # ~419M elements - Extra large size
]

if __name__ == "__main__":
    print("Starting matrix multiplication comparison experiments...")
    print("=" * 80)
    
    print("Test Matrices:")
    for m1, m2, m3 in test_matrices:
        elements = m1 * m2 * m3
        print(f"  {m1}x{m2} * {m2}x{m3} (~{elements/1e6:.1f}M elements)")
    
    print("\nBlock Configurations:")
    print(f"  {BLOCK_SIZE}x{BLOCK_SIZE} ({BLOCK_SIZE*BLOCK_SIZE} threads)")
    
    # Run experiments
    results_df = run_experiments()
    
    # Display results
    visualize_results(results_df)
    
    print("\nExperiments completed successfully!")
    print("Results have been saved to 'gpu_experiment_results.csv'")