import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run_scale_test(label: str, dims: int, heads: int):
    print(f"\n[*] Evaluating Scale: {label} (Dim: {dims}, Heads: {heads})")
    
    agi = JuniorAGI(dims=dims, heads=heads)
    local_dims = agi.mesh.shard_dimension(dims)
    x = mx.random.normal((1, 64, local_dims)) 
    
    # Warmup
    warmup_out = agi.forward(x)
    mx.eval(warmup_out["y"]) 
    pb_used = warmup_out["metrics"]["power_budget"]
    therm = warmup_out["metrics"]["thermal_pressure"]
    
    start = time.perf_counter()
    for _ in range(10):
        mx.eval(agi.forward(x)["y"])
    end = time.perf_counter()
    
    mem_used = mx.get_active_memory() / 1024**2
    
    q_max = max(3.0, min(127.0, round((pb_used * 120.0) + 7.0)))
    
    # Synthetic Joules Per Inference (JPI) Proxy
    # Higher thermal pressure and higher dimension = more Joules
    jpi_proxy = (dims * therm * (end-start)) / 1000.0
    
    print(f"  -> Power Budget : {pb_used:.2f} (Routing max_val={int(q_max)} Activation)")
    print(f"  -> Thermal Load : {therm:.2f} (JPI Proxy: {jpi_proxy:.3f} J)")
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
