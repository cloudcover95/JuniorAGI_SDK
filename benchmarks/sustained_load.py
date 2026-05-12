# benchmarks/sustained_load.py
import time, sys, os, mlx.core as mx
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI
from infrastructure.hardware_matrix import HardwareMatrix

def run_sustained_test(duration_sec: int = 60, scale_preset: str = "7B"):
    hw = HardwareMatrix()
    print(f"\n[*] Initiating Thermodynamic Sustained Load Test: {duration_sec} Seconds")
    print(f"[*] Target Scale: {scale_preset} | Hardware: {hw.chip_info}")
    
    agi = JuniorAGI(target_scale=scale_preset)
    local_dims = agi.mesh.shard_dimension(agi.MODEL_PRESETS[scale_preset]["dims"])
    x = mx.random.normal((1, 128, local_dims))
    
    mx.eval(agi.forward(x)["y"]) # Warmup
    
    start_time = time.perf_counter()
    inferences = 0
    total_jpi = 0.0
    
    print(f"{'Time (s)':<10} | {'TPS':<8} | {'Pwr Budg':<10} | {'Thermal%':<10} | {'JPI (Joules)':<15} | {'VRAM (MB)':<10}")
    print("-" * 75)
    
    while (time.perf_counter() - start_time) < duration_sec:
        loop_start = time.perf_counter()
        
        out = agi.forward(x)
        mx.eval(out["y"])
        
        loop_end = time.perf_counter()
        inferences += 1
        total_jpi += out["jpi"]
        
        # Report every 5 inferences
        if inferences % 5 == 0:
            elapsed = loop_end - start_time
            tps = inferences / elapsed
            mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
            mem = mem_func() / 1024**2
            pb = out["metrics"]["power_budget"]
            therm = out["metrics"]["thermal_pressure"] * 100.0
            print(f"{elapsed:<10.1f} | {tps:<8.2f} | {pb:<10.3f} | {therm:<9.1f}% | {out['jpi']:<15.4f} | {mem:<10.1f}")
            
    print("-" * 75)
    print(f"Total Inferences : {inferences}")
    print(f"Total Energy     : {total_jpi:.2f} Joules")
    print(f"Avg JPI          : {total_jpi/inferences:.4f} Joules/Inference")

if __name__ == "__main__":
    run_sustained_test(duration_sec=20, scale_preset="7B")
