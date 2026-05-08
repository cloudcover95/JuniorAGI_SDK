import time, sys, os, mlx.core as mx
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run():
    print("[*] JuniorAGI v0.71.0 Base Benchmark (7B Equivalent, Dim: 4096)...")
    agi = JuniorAGI(dims=4096, heads=32)
    
    # Inject 3D Tensor: (Batch=1, Seq=128, Dim=4096)
    x = mx.random.normal((1, 128, 4096))
    
    _ = agi.forward(x) # Compile and Warm
    
    start = time.perf_counter()
    for _ in range(50):
        out = agi.forward(x)
        mx.eval(out["y"])
        
    end = time.perf_counter()
    mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
    mem_mb = mem_func() / 1024**2
    
    print("-" * 50)
    print(f"Substrate    : v0.71.0 (7B Block)")
    print(f"TPS          : {50/(end-start):.2f} Blocks/sec")
    print(f"Memory Shift : Context array scaled to {out['context_shape']}")
    print(f"UMA VRAM     : {mem_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__": run()
