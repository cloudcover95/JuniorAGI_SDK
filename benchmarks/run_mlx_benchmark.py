# benchmarks/run_mlx_benchmark.py
import time
import sys
import os
import mlx.core as mx

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from inference.bitnet_layers import DynamicBitLinear
from manifolds.spectral_tda import SpectralTDAManifold

def execute_benchmark(iterations: int = 100):
    print(f"[*] Initiating JuniorAGI MLX Native Benchmark ({iterations} iterations)...")
    
    # Initialize components
    layer = DynamicBitLinear(4096, 4096, max_rank=32)
    tda = SpectralTDAManifold()
    
    # Simulate high-dimensional input stream
    x = mx.random.normal((128, 4096))
    
    # Pre-Warm MPS Graph
    mx.eval(layer(x, tau=0.1))
    
    start_mem = mx.metal.get_active_memory() / 1024**2
    start_time = time.perf_counter()
    
    for i in range(iterations):
        # 1. Forward Pass (BitNet + Residual)
        y = layer(x, tau=0.08)
        
        # 2. Extract Topological Proxy (SVD Spectral Gaps)
        topology = tda.extract_betti_proxies(y)
        mx.eval(y)
        
    end_time = time.perf_counter()
    end_mem = mx.metal.get_active_memory() / 1024**2
    
    total_time = end_time - start_time
    ops_per_sec = iterations / total_time
    
    print("-" * 50)
    print(f"Hardware        : Apple Silicon (MPS: {mx.metal.is_available()})")
    print(f"Total Time      : {total_time:.4f} seconds")
    print(f"Throughput      : {ops_per_sec:.2f} inferences/sec")
    print(f"Active Memory   : {start_mem:.2f} MB -> {end_mem:.2f} MB (Delta: {end_mem-start_mem:.2f} MB)")
    print(f"Topology Sample : Betti-0: {topology['betti_0_proxy']}, Betti-1: {topology['betti_1_proxy']}")
    print("-" * 50)

if __name__ == "__main__":
    execute_benchmark(100)
