import time, sys, os, mlx.core as mx
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run():
    print("[*] JuniorAGI v0.73.0 Base Benchmark (7B Equivalent, Dim: 4096)...")
    agi = JuniorAGI(dims=4096, heads=32)
    x = mx.random.normal((1, 128, 4096))
    
    _ = agi.forward(x) 
    
    start = time.perf_counter()
    for _ in range(50):
        out = agi.forward(x)
        mx.eval(out["y"])
        
    end = time.perf_counter()
    mem_mb = mx.get_active_memory() / 1024**2
    
    print("-" * 50)
    print(f"Substrate    : v0.73.0 (Proactive C2V)")
    print(f"TPS          : {50/(end-start):.2f} Blocks/sec")
    print(f"Power Budget : {out['power_budget']:.2f}")
    print(f"UMA VRAM     : {mem_mb:.2f} MB")
    print("-" * 50)

if __name__ == "__main__": run()
