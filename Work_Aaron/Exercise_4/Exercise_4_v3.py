import cupy as cp
import numpy as np
import time
from queue import Queue, Empty
import threading

class PipelinedMatrixProcessor:
    def __init__(self, matrix_size=1000, num_buffers=3):
        self.matrix_size = matrix_size
        self.num_buffers = num_buffers
        
        # Create streams
        self.h2d_stream = cp.cuda.Stream(non_blocking=True)
        self.compute_streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        self.d2h_stream = cp.cuda.Stream(non_blocking=True)
        
        # Create events for synchronization
        self.h2d_events = [cp.cuda.Event() for _ in range(num_buffers)]
        self.compute_events = [cp.cuda.Event() for _ in range(num_buffers)]
        
        # Initialize buffer pools
        self.init_buffers()
        
        # Create queues for buffer management
        self.free_input_buffers = Queue()
        self.ready_for_compute = Queue()
        self.ready_for_d2h = Queue()
        self.free_output_buffers = Queue()
        
        # Fill buffer queues
        for i in range(num_buffers):
            self.free_input_buffers.put(i)
            self.free_output_buffers.put(i)
        
        self.running = True
        
    def init_buffers(self):
        # Host buffers
        self.host_input_buffers = [
            (np.empty((self.matrix_size, self.matrix_size), dtype=np.float32),
             np.empty((self.matrix_size, self.matrix_size), dtype=np.float32))
            for _ in range(self.num_buffers)
        ]
        self.host_output_buffers = [
            np.empty((self.matrix_size, self.matrix_size), dtype=np.float32)
            for _ in range(self.num_buffers)
        ]
        
        # Device buffers
        self.dev_input_buffers = [
            (cp.empty((self.matrix_size, self.matrix_size), dtype=cp.float32),
             cp.empty((self.matrix_size, self.matrix_size), dtype=cp.float32))
            for _ in range(self.num_buffers)
        ]
        self.dev_output_buffers = [
            cp.empty((self.matrix_size, self.matrix_size), dtype=cp.float32)
            for _ in range(self.num_buffers)
        ]

    def h2d_worker(self):
        """Continuously transfers data from host to device"""
        while self.running:
            try:
                # Get a free input buffer
                buf_idx = self.free_input_buffers.get(timeout=1.0)
                
                # Generate new input matrices
                self.host_input_buffers[buf_idx] = (
                    np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32),
                    np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
                )
                
                with self.h2d_stream:
                    # Transfer to device
                    cp.copyto(self.dev_input_buffers[buf_idx][0], 
                            cp.asarray(self.host_input_buffers[buf_idx][0]))
                    cp.copyto(self.dev_input_buffers[buf_idx][1], 
                            cp.asarray(self.host_input_buffers[buf_idx][1]))
                    
                    # Record event
                    self.h2d_events[buf_idx].record(self.h2d_stream)
                
                # Signal compute stage
                self.ready_for_compute.put(buf_idx)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in H2D worker: {e}")
                continue

    def compute_worker(self, stream_idx):
        """Performs matrix multiplication"""    
        compute_stream = self.compute_streams[stream_idx]
        
        while self.running:
            try:
                # Get a buffer ready for computation
                buf_idx = self.ready_for_compute.get(timeout=1.0)
                
                with compute_stream:
                    # Wait for H2D to complete
                    self.h2d_events[buf_idx].synchronize()
                    
                    # Perform multiplication using the current stream context
                    result = cp.matmul(self.dev_input_buffers[buf_idx][0],
                                     self.dev_input_buffers[buf_idx][1],
                                     out=self.dev_output_buffers[buf_idx])
                    
                    # Record completion
                    self.compute_events[buf_idx].record(compute_stream)
                
                # Signal D2H stage
                self.ready_for_d2h.put(buf_idx)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in compute worker {stream_idx}: {e}")
                continue

    def d2h_worker(self):
        """Continuously transfers results back to host"""
        results_counter = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Get a buffer ready for transfer back to host
                buf_idx = self.ready_for_d2h.get(timeout=1.0)
                
                with self.d2h_stream:
                    # Wait for computation to complete
                    self.compute_events[buf_idx].synchronize()
                    
                    # Transfer result back to host
                    self.dev_output_buffers[buf_idx].get(
                        out=self.host_output_buffers[buf_idx],
                        stream=self.d2h_stream
                    )
                
                results_counter += 1
                if results_counter % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {results_counter} matrices. "
                          f"Average time per matrix: {elapsed/results_counter:.3f} seconds")
                
                # Return buffers to free pools
                self.free_input_buffers.put(buf_idx)
                self.free_output_buffers.put(buf_idx)
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in D2H worker: {e}")
                continue

    def start_processing(self):
        """Starts all worker threads"""
        self.threads = [
            threading.Thread(target=self.h2d_worker),
            threading.Thread(target=self.compute_worker, args=(0,)),
            threading.Thread(target=self.compute_worker, args=(1,)),
            threading.Thread(target=self.d2h_worker)
        ]
        
        for thread in self.threads:
            thread.start()

    def stop_processing(self):
        """Stops all worker threads"""
        self.running = False
        for thread in self.threads:
            thread.join()

def main():
    # Create and start processor
    processor = PipelinedMatrixProcessor(matrix_size=1000, num_buffers=3)
    
    try:
        processor.start_processing()
        # Let it run for a while
        time.sleep(30)
    finally:
        processor.stop_processing()

if __name__ == "__main__":
    main()