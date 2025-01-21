import numpy as np
import cupy as cp
import threading
from queue import Queue, Empty
import time
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')

class GPUWorker:
    def __init__(self, input_queue: Queue, compute_queue: Queue, output_queue: Queue, 
                 result_queue: Queue, worker_id: int):
        self.input_queue = input_queue      # Queue for incoming CPU matrices
        self.compute_queue = compute_queue  # Queue for matrices ready for GPU computation
        self.output_queue = output_queue    # Queue for GPU results ready for transfer back
        self.result_queue = result_queue    # Queue for final results
        self.worker_id = worker_id
        
        # Create separate streams for memory transfers and computation
        self.transfer_stream = cp.cuda.Stream(non_blocking=True)
        self.compute_stream = cp.cuda.Stream(non_blocking=True)
        self.running = True
        
        # Create events for synchronization
        self.h2d_event = cp.cuda.Event()
        self.compute_event = cp.cuda.Event()
        
    def transfer_to_device(self):
        """Handle host to device transfers"""
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                if item is None:
                    self.running = False
                    self.compute_queue.put(None)  # Signal compute thread to stop
                    break
                
                matrix_id, A, B = item
                with self.transfer_stream:
                    # Transfer matrices to GPU
                    A_gpu = cp.asarray(A)
                    B_gpu = cp.asarray(B)
                    self.h2d_event.record(self.transfer_stream)
                
                # Put matrices in compute queue with their events
                self.compute_queue.put((matrix_id, A_gpu, B_gpu, self.h2d_event))
                self.input_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in transfer to device: {e}")
                
    def compute(self):
        """Handle GPU computation"""
        while self.running:
            try:
                item = self.compute_queue.get(timeout=1.0)
                if item is None:
                    self.running = False
                    self.output_queue.put(None)  # Signal transfer back thread to stop
                    break
                
                matrix_id, A_gpu, B_gpu, h2d_event = item
                
                with self.compute_stream:
                    # Wait for transfers to complete
                    h2d_event.synchronize()
                    
                    # Perform computation
                    start_time = time.perf_counter()
                    C_gpu = cp.matmul(A_gpu, B_gpu)
                    self.compute_event.record(self.compute_stream)
                    
                    duration = time.perf_counter() - start_time
                
                # Queue result for transfer back to host
                self.output_queue.put((matrix_id, C_gpu, duration, self.compute_event))
                self.compute_queue.task_done()
                
                # Clean up input matrices
                del A_gpu, B_gpu
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in compute: {e}")
                
    def transfer_to_host(self):
        """Handle device to host transfers"""
        while self.running:
            try:
                item = self.output_queue.get(timeout=1.0)
                if item is None:
                    self.running = False
                    break
                    
                matrix_id, C_gpu, compute_duration, compute_event = item
                
                with self.transfer_stream:
                    # Wait for computation to complete
                    compute_event.synchronize()
                    
                    # Transfer result back to host
                    start_time = time.perf_counter()
                    C = cp.asnumpy(C_gpu)
                    transfer_duration = time.perf_counter() - start_time
                
                # Clean up GPU memory
                del C_gpu
                
                # Put final result in result queue
                self.result_queue.put((
                    matrix_id, C, compute_duration, 
                    transfer_duration, self.worker_id
                ))
                self.output_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in transfer to host: {e}")

class MatrixMultiplier:
    def __init__(self, num_workers: int = 3):
        self.input_queue = Queue()
        self.compute_queues = [Queue() for _ in range(num_workers)]
        self.output_queues = [Queue() for _ in range(num_workers)]
        self.result_queue = Queue()
        self.num_workers = num_workers
        self.workers: List[Dict[str, threading.Thread]] = []
        self.results = {}

    def start_workers(self):
        for i in range(self.num_workers):
            gpu_worker = GPUWorker(
                self.input_queue, 
                self.compute_queues[i],
                self.output_queues[i], 
                self.result_queue,
                i
            )
            
            # Create three threads per worker for the pipeline stages
            h2d_thread = threading.Thread(
                target=gpu_worker.transfer_to_device,
                name=f"H2D-Worker-{i}"
            )
            compute_thread = threading.Thread(
                target=gpu_worker.compute,
                name=f"Compute-Worker-{i}"
            )
            d2h_thread = threading.Thread(
                target=gpu_worker.transfer_to_host,
                name=f"D2H-Worker-{i}"
            )
            
            # Start all threads
            h2d_thread.start()
            compute_thread.start()
            d2h_thread.start()
            
            # Store threads
            self.workers.append({
                'h2d': h2d_thread,
                'compute': compute_thread,
                'd2h': d2h_thread
            })

    def stop_workers(self):
        # Send stop signal to input queues
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        
        # Wait for all threads to finish
        for worker_threads in self.workers:
            for thread in worker_threads.values():
                thread.join()
            
        logging.info("All workers terminated")

    def add_matrices(self, matrix_id: int, A: np.ndarray, B: np.ndarray):
        self.input_queue.put((matrix_id, A, B))

    def collect_results(self) -> dict:
        while not self.result_queue.empty():
            matrix_id, result, compute_time, transfer_time, worker_id = self.result_queue.get()
            self.results[matrix_id] = {
                'result': result,
                'compute_time': compute_time,
                'transfer_time': transfer_time,
                'total_time': compute_time + transfer_time,
                'worker_id': worker_id
            }
        return self.results

def run_benchmark(matrix_sizes: List[Tuple[int, int, int]], num_workers: int = 3):
    multiplier = MatrixMultiplier(num_workers)
    multiplier.start_workers()

    try:
        # Generate and queue test matrices
        for idx, (M, K, N) in enumerate(matrix_sizes):
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            multiplier.add_matrices(idx, A, B)
            logging.info(f"Matrix {idx} ({M}x{K} @ {K}x{N}) added to queue")

        # Wait for all matrices to be processed
        multiplier.input_queue.join()
        for q in multiplier.compute_queues:
            q.join()
        for q in multiplier.output_queues:
            q.join()
        
        # Collect and display results
        results = multiplier.collect_results()
        
        for idx, (M, K, N) in enumerate(matrix_sizes):
            if idx in results:
                r = results[idx]

    finally:
        # Ensure workers are stopped
        multiplier.stop_workers()

if __name__ == "__main__":
    # Example benchmark configuration
    matrix_sizes = [
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
        (4096, 4096, 4096),
    ]
    
    run_benchmark(matrix_sizes, num_workers=1)