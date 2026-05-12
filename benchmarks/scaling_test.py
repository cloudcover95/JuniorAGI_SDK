import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI
from infrastructure.hardware_matrix import HardwareMatrix

def run_preset_test(scale_preset: str):
    print(f"\n[*] Evaluating Omni-Tier: {scale_preset}")
    
    agi = JuniorAGI(target_scale=scale_preset)
    
    # Retrieve local shard dimensions
    dims = agi.MODEL_PRESETS[scale_preset]["dims"]
    local_dims = agi.mesh.shard_dimension(dims)
    x = mx.random.normal((1, 32, local_dims)) 
    
    # Warmup
    warmup_out = agi.forward(x)
    mx.eval(warmup_out["y"]) 
    pb_used = warmup_out["metrics"]["power_budget"]
    therm = warmup_out["metrics"]["thermal_pressure"]
    
    start = time.perf_counter()
    for _ in range(5):
        mx.eval(agi.forward(x)["y"])
    end = time.perf_counter()
    
    q_max = max(3.0, min(127.0, round((pb_used * 120.0) + 7.0)))
    
    print(f"  -> Power Budget : {pb_used:.2f} (Routing max_val={int(q_max)})")
    print(f"  -> Thermal Load : {therm:.2f} (Avg JPI: {warmup_out['jpi']:.3f} J)")
    print(f"  -> Throughput   : {5/(end-start):.2f} Blocks/sec")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    hw = HardwareMatrix()
    print(f"=== JuniorAGI Substrate Omni-Tier Scaling ===")
    print(f"Detected Hardware : {hw.chip_info} | UMA: {hw.total_mem}GB | MPS: {hw.mps_available}")
    
    run_preset_test("7B")
    run_preset_test("70B")
    run_preset_test("100B")
    print("=============================================")
