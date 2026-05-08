# benchmarks/sustained_load.py
import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI
import warnings
warnings.filterwarnings("ignore")

def run_sustained_test(duration_sec: int = 15):
    print(f"\n[*] Initiating Sustained Thermal Durability Test ({duration_sec} Seconds)")
    print("[*] Simulating a 4-Layer 7B-Class Manifold...")
    
    agi = JuniorAGI(dims=4096, heads=32, num_layers=4)
    local_dims = agi.mesh.shard_dimension(4096)
    x = mx.random.normal((1, 64, local_dims))
    
    mx.eval(agi.forward(x)["y"]) # Warmup
    
    start_time = time.perf_counter()
    inferences = 0
    total_jpi = 0.0
    
    print(f"{'Time (s)':<10} | {'TPS':<10} | {'Power Budg':<12} | {'JPI (Joules)':<15} | {'VRAM (MB)':<10}")
    print("-" * 65)
    
    while (time.perf_counter() - start_time) < duration_sec:
        loop_start = time.perf_counter()
        
        out = agi.forward(x)
        mx.eval(out["y"])
        
        loop_end = time.perf_counter()
        inferences += 1
        total_jpi += out["jpi"]
        
        if inferences % 5 == 0:
            elapsed = loop_end - start_time
            tps = inferences / elapsed
            mem = mx.get_active_memory() / 1024**2
            pb = out["metrics"]["power_budget"]
            print(f"{elapsed:<10.1f} | {tps:<10.2f} | {pb:<12.3f} | {out['jpi']:<15.4f} | {mem:<10.1f}")
            
    print("-" * 65)
    print(f"Total Inferences : {inferences}")
    print(f"Total Energy     : {total_jpi:.2f} Joules")
    print(f"Avg JPI          : {total_jpi/inferences:.4f} Joules/Inference")

if __name__ == "__main__":
    run_sustained_test(15)
