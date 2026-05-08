import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run_scale_test(label: str, dims: int, heads: int):
    print(f"\n[*] Evaluating Scale: {label} (Dim: {dims}, Heads: {heads})")
    agi = JuniorAGI(dims=dims, heads=heads)
    x = mx.random.normal((1, 64, dims)) # Standard short context for structural test
    
    mx.eval(agi.forward(x)["y"]) # Warmup
    
    start = time.perf_counter()
    for _ in range(10):
        mx.eval(agi.forward(x)["y"])
    end = time.perf_counter()
    
    mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
    print(f"  -> Throughput: {10/(end-start):.2f} Blocks/sec")
    print(f"  -> UMA VRAM  : {mem_func() / 1024**2:.2f} MB")

if __name__ == "__main__":
    print("=== JuniorAGI Substrate Scaling Array ===")
    run_scale_test("Phase 1 (7B-Class)", 4096, 32)
    run_scale_test("Phase 2 (70B-Class / Dual-Die)", 8192, 64)
    run_scale_test("Phase 3 (100B-Class / Thunderbolt Mesh)", 12288, 96)
    print("=========================================")
