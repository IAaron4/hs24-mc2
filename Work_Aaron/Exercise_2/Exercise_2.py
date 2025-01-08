import cupy as cp
import numpy as np
from numba import cuda, float32
import math
import time
from typing import Dict, List, Tuple
import pandas as pd

# Test matrices configurations (M, K, N) for multiplication of shape (M,K) * (K,N)
test_matrices = [
    (5120, 5120, 5120),     # ~26M elements
    (8192, 4096, 8192),     # ~34M elements
    (10240, 10240, 10240),  # ~105M elements
]

# Block configurations to test (threads_x, threads_y)
block_configs = [
    (8, 8),     # 64 threads
    (16, 16),   # 256 threads
    (32, 16),   # 512 threads
    (32, 32)    # 1024 threads
]

def cpu_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """CPU implementation for comparison"""
    return np.matmul(A, B)

def run_gpu_experiment(A: np.ndarray, B: np.ndarray, block_size: Tuple[int, int], 
                      num_warmup: int = 3, num_runs: int = 3) -> Dict:
    """Run GPU experiment with specific block size including warmup runs"""
    TPB_x, TPB_y = block_size
    M, K = A.shape
    K, N = B.shape
    
    # Copy to device
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.device_array((M, N), dtype=np.float32)

    # Configure grid
    threadsperblock = block_size
    blockspergrid_x = math.ceil(M / TPB_x)
    blockspergrid_y = math.ceil(N / TPB_y)
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Warmup runs
    print(f"Performing {num_warmup} warmup runs...")
    for _ in range(num_warmup):
        fast_matmul[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu, TPB_x, TPB_y)
        cuda.synchronize()

    # Measurement runs
    times = []
    print(f"Performing {num_runs} measurement runs...")
    for i in range(num_runs):
        # Clear cache and synchronize
        cuda.synchronize()
        
        start = time.perf_counter()
        fast_matmul[blockspergrid, threadsperblock](A_gpu, B_gpu, C_gpu, TPB_x, TPB_y)
        cuda.synchronize()
        end = time.perf_counter()
        
        execution_time = end - start
        times.append(execution_time)
        print(f"Run {i+1}: {execution_time:.4f} seconds")

    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

@cuda.jit
def fast_matmul(A, B, C, TPB_x, TPB_y):
    """CUDA kernel for matrix multiplication using shared memory"""
    # Define shared memory arrays
    sA = cuda.shared.array(shape=(32, 32), dtype=float32)  # Using maximum possible size
    sB = cuda.shared.array(shape=(32, 32), dtype=float32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    # Use actual block size for calculations
    TPB = min(TPB_x, TPB_y)

    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx+i*TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty+i*TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        cuda.syncthreads()

    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

def run_experiments(save_results: bool = True):
    results = []
    
    for m1, m2, m3 in test_matrices:
        print(f"\nTesting matrix size: {m1}x{m2} * {m2}x{m3}")
        A = np.random.randn(m1, m2).astype(np.float32)
        B = np.random.randn(m2, m3).astype(np.float32)

        # CPU baseline
        print("\nRunning CPU baseline...")
        start = time.perf_counter()
        cpu_matmul(A, B)
        cpu_time = time.perf_counter() - start
        print(f"CPU Time: {cpu_time:.3f}s")

        # GPU experiments with different block sizes
        for block_size in block_configs:
            print(f"\nTesting block size: {block_size[0]}x{block_size[1]}")
            gpu_results = run_gpu_experiment(A, B, block_size)
            speedup = cpu_time / gpu_results['mean']
            
            results.append({
                'Matrix Size': f"{m1}x{m2}x{m3}",
                'Block Size': f"{block_size[0]}x{block_size[1]}",
                'CPU Time': cpu_time,
                'GPU Time Mean': gpu_results['mean'],
                'GPU Time Std': gpu_results['std'],
                'Speedup': speedup,
                'Elements': m1 * m2 * m3
            })
            
            print(f"Average GPU Time: {gpu_results['mean']:.4f}s (Speedup: {speedup:.2f}x)")
            print(f"Std Dev: {gpu_results['std']:.4f}s")

    results_df = pd.DataFrame(results)
    
    if save_results:
        results_df.to_csv('gpu_experiment_results.csv', index=False)
        
    return results_df

def visualize_results(df: pd.DataFrame):
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("=" * 80)
    stats = df.groupby(['Matrix Size', 'Block Size']).agg({
        'GPU Time Mean': ['mean', 'std'],
        'Speedup': ['mean', 'std']
    }).round(4)
    
    print(stats)
    
if __name__ == "__main__":
    results_df = run_experiments()
    visualize_results(results_df)