import numpy as np
from typing import Dict, List
import pandas as pd
from IPython.display import display

# Import dependencies from previous implentations
from GPUTimer import GPUTimer
from Test_Cases import (
    cupy_test_cases,
    numba_test_cases
)

def run_benchmarks(backend: str = 'cupy') -> Dict[str, List[float]]:
    """Run all test cases for specified backend"""
    timer = GPUTimer(backend=backend)
    results = {}
    
    # Select test cases based on backend
    test_cases = cupy_test_cases if backend == 'cupy' else numba_test_cases
    
    print(f"\nRunning {backend.upper()} benchmarks:")
    print("=" * 50)
    
    for name, func in test_cases.items():
        print(f"\nExecuting {name}")
        measurements = timer.measure_execution(func)
        results[name] = measurements
        
    return results

def visualize_comparison(cupy_results: Dict[str, List[float]], 
                        numba_results: Dict[str, List[float]]):
    """Create comparative visualizations of the results"""
    
    # Prepare data for plotting
    data = []
    for name in cupy_results.keys():
        # CuPy data
        for measurement in cupy_results[name]:
            data.append({
                'Test Case': name,
                'Time (ms)': measurement,
                'Backend': 'CuPy'
            })
        # Numba data
        for measurement in numba_results[name]:
            data.append({
                'Test Case': name,
                'Time (ms)': measurement,
                'Backend': 'Numba'
            })
    
    df = pd.DataFrame(data)
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("=" * 80)
    stats_data = []
    
    for name in cupy_results.keys():
        cupy_mean = np.mean(cupy_results[name])
        cupy_std = np.std(cupy_results[name])
        numba_mean = np.mean(numba_results[name])
        numba_std = np.std(numba_results[name])
        
        stats_data.append({
            'Test Case': name,
            'CuPy Mean (ms)': f"{cupy_mean:.2f} ± {cupy_std:.2f}",
            'Numba Mean (ms)': f"{numba_mean:.2f} ± {numba_std:.2f}",
            'Speedup (CuPy/Numba)': f"{numba_mean/cupy_mean:.2f}x"
        })
    
    stats_df = pd.DataFrame(stats_data)
    display(stats_df)

def main():
    # Run benchmarks for both backends
    print("Starting benchmarks...\n")
    
    print("Running CuPy benchmarks...")
    cupy_results = run_benchmarks(backend='cupy')
    
    print("\nRunning Numba benchmarks...")
    numba_results = run_benchmarks(backend='numba')
    
    # Visualize results
    print("\nGenerating comparative visualization...")
    visualize_comparison(cupy_results, numba_results)

if __name__ == "__main__":
    main()