import cupy as cp
import numpy as np
from numba import cuda, float32
import math

# =============== CuPy Implementation ===============
def cupy_test_case_1():
    """Small matrices: 128x128 * 128x128"""
    x = cp.random.randn(128, 128, dtype=cp.float32)
    y = cp.random.randn(128, 128, dtype=cp.float32)
    return cp.matmul(x, y)

def cupy_test_case_2():
    """Medium-small matrices: 512x256 * 256x512"""
    x = cp.random.randn(512, 256, dtype=cp.float32)
    y = cp.random.randn(256, 512, dtype=cp.float32)
    return cp.matmul(x, y)

def cupy_test_case_3():
    """Medium matrices: 1024x1024 * 1024x1024"""
    x = cp.random.randn(1024, 1024, dtype=cp.float32)
    y = cp.random.randn(1024, 1024, dtype=cp.float32)
    return cp.matmul(x, y)

def cupy_test_case_4():
    """Large rectangular matrices: 2048x1536 * 1536x512"""
    x = cp.random.randn(2048, 1536, dtype=cp.float32)
    y = cp.random.randn(1536, 512, dtype=cp.float32)
    return cp.matmul(x, y)

def cupy_test_case_5():
    """Maximum size matrices: 4000x4000 * 4000x4000"""
    x = cp.random.randn(4000, 4000, dtype=cp.float32)
    y = cp.random.randn(4000, 4000, dtype=cp.float32)
    return cp.matmul(x, y)

# Dictionary for CuPy test cases
cupy_test_cases = {
    "Small (128x128)": cupy_test_case_1,
    "Medium-small (512x256 * 256x512)": cupy_test_case_2,
    "Medium (1024x1024)": cupy_test_case_3,
    "Large rectangular (2048x1536 * 1536x512)": cupy_test_case_4,
    "Maximum (4000x4000)": cupy_test_case_5
}

# =============== Numba Implementation ===============
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx+i*TPB) < A.shape[1]:
          sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty+i*TPB) < B.shape[0]:
          sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

def numba_matrix_multiply(A, B):
    """Helper function to set up and execute Numba CUDA matrix multiplication"""
    M, K = A.shape
    K, N = B.shape
    
    # Copy arrays to device
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    
    # Allocate memory on GPU for result
    C_global_mem = cuda.device_array((M, N), dtype=np.float32)

    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(M / TPB)
    blockspergrid_y = math.ceil(N / TPB)
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Start the kernel
    fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    
    # Copy result back to host
    C = C_global_mem.copy_to_host()

    return C

def numba_test_case_1():
    """Small matrices: 128x128 * 128x128"""
    A = np.random.randn(128, 128).astype(np.float32)
    B = np.random.randn(128, 128).astype(np.float32)
    return numba_matrix_multiply(A, B)

def numba_test_case_2():
    """Medium-small matrices: 512x256 * 256x512"""
    A = np.random.randn(512, 256).astype(np.float32)
    B = np.random.randn(256, 512).astype(np.float32)
    return numba_matrix_multiply(A, B)

def numba_test_case_3():
    """Medium matrices: 1024x1024 * 1024x1024"""
    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)
    return numba_matrix_multiply(A, B)

def numba_test_case_4():
    """Large rectangular matrices: 2048x1536 * 1536x512"""
    A = np.random.randn(2048, 1536).astype(np.float32)
    B = np.random.randn(1536, 512).astype(np.float32)
    return numba_matrix_multiply(A, B)

def numba_test_case_5():
    """Maximum size matrices: 4000x4000 * 4000x4000"""
    A = np.random.randn(4000, 4000).astype(np.float32)
    B = np.random.randn(4000, 4000).astype(np.float32)
    return numba_matrix_multiply(A, B)

# Dictionary for Numba test cases
numba_test_cases = {
    "Small (128x128)": numba_test_case_1,
    "Medium-small (512x256 * 256x512)": numba_test_case_2,
    "Medium (1024x1024)": numba_test_case_3,
    "Large rectangular (2048x1536 * 1536x512)": numba_test_case_4,
    "Maximum (4000x4000)": numba_test_case_5
}