import time, sys, os, mlx.core as mx
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from kernel.agi_kernel import JuniorAGI

def run(sec: int = 15):
    print(f"[*] v0.81.0 Sustained Load / VRAM Lockdown Test ({sec}s)")
    agi = JuniorAGI()
    x = mx.random.normal((1, 64, agi.PRESETS["7B"][0]))
    mx.eval(agi.forward(x)["y"])
    
    t0 = time.perf_counter()
    inf, jpi = 0, 0.0
    print(f"{'Time':<8} | {'TPS':<6} | {'PB':<6} | {'Therm%':<8} | {'JPI':<8} | {'VRAM(MB)'}")
    print("-" * 55)
    
    while (time.perf_counter() - t0) < sec:
        out = agi.forward(x)
        inf += 1
        jpi += out["jpi"]
        if inf % 5 == 0:
            e = time.perf_counter() - t0
            mem = mx.get_active_memory() / 1024**2 if hasattr(mx, 'get_active_memory') else 0
            pb, th = out["metrics"]["power_budget"], out["metrics"]["thermal_pressure"] * 100
            print(f"{e:<8.1f} | {inf/e:<6.2f} | {pb:<6.2f} | {th:<8.1f} | {out['jpi']:<8.2f} | {mem:.1f}")
            
if __name__ == "__main__": run()
