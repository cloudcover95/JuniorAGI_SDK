import time, sys, os, mlx.core as mx
import warnings

# Suppress standard MLX deprecation warnings for clean output
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run():
    print("[*] JuniorAGI v0.68.0 Authentic MLX Benchmark...")
    agi = JuniorAGI()
    x = mx.random.normal((64, 1024))
    
    # Compile pre-warm
    _ = agi.forward(x)
    
    start = time.perf_counter()
    for _ in range(100):
        # Full pass: Ternary BitNet + SVD Gram Matrix Spectral TDA
        out = agi.forward(x)
        mx.eval(out["y"])
        
    end = time.perf_counter()
    
    # Modern MLX memory lookup
    mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
    mem_mb = mem_func() / 1024**2
    
    print("-" * 50)
    print(f"Substrate   : v0.68.0 Authentic")
    print(f"TPS         : {100/(end-start):.2f} inferences/sec")
    print(f"Topology    : B0: {out['topology']['betti_0']} | B1: {out['topology']['betti_1']}")
    print(f"UMA VRAM    : {mem_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__": run()
