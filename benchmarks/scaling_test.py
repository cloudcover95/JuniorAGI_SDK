# benchmarks/scaling_test.py
import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run_scale_test(label: str, dims: int, heads: int):
    print(f"\n[*] Evaluating Scale: {label} (Dim: {dims}, Heads: {heads})")
    
    # Initialize target topology
    agi = JuniorAGI(dims=dims, heads=heads)
    x = mx.random.normal((1, 64, dims)) 
    
    # Pre-warm Metal Shaders
    warmup = agi.forward(x)
    mx.eval(warmup["y"]) 
    pb_used = warmup["power_budget"]
    
    start = time.perf_counter()
    for _ in range(10):
        mx.eval(agi.forward(x)["y"])
    end = time.perf_counter()
    
    mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
    mem_used = mem_func() / 1024**2
    
    bit_width = 8 if pb_used > 0.7 else 6
    
    print(f"  -> Power Budget : {pb_used:.2f} (Routing {bit_width}-bit Activation)")
    print(f"  -> Throughput   : {10/(end-start):.2f} Blocks/sec")
    print(f"  -> UMA VRAM     : {mem_used:.2f} MB")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("=== JuniorAGI Substrate Scaling Array ===")
    run_scale_test("Phase 1 (7B-Class / Single Die)", 4096, 32)
    run_scale_test("Phase 2 (70B-Class / Dual-Die UltraFusion)", 8192, 64)
    run_scale_test("Phase 3 (100B-Class / Distributed Mesh)", 12288, 96)
    print("=========================================")
