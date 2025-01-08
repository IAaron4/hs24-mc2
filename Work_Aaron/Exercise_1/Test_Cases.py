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
@cuda.jit
def matmul_kernel(A, B, C):
    """CUDA kernel for matrix multiplication"""
    row, col = cuda.grid(2)
    
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

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
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(M / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Start the kernel
    matmul_kernel[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    
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