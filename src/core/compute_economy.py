# src/core/compute_economy.py
import psutil
import mlx.core as mx
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()

    def get_c2v_metrics(self) -> dict:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        cpu_headroom = max(0.1, 1.0 - cpu_load)
        
        # MLX Deprecation Fix: mx.get_active_memory()
        vram_used = (mx.get_active_memory() / 1024**3)
        uma_headroom = max(0.1, 1.0 - (vram_used / self.specs['uma_gb']))
        
        # Power budget modulates bit-width: [0.0 - 1.0]
        power_budget = (cpu_headroom * 0.4) + (uma_headroom * 0.6)
        
        return {
            "power_budget": round(power_budget, 4),
            "cpu_headroom": round(cpu_headroom, 4),
            "uma_headroom": round(uma_headroom, 4)
        }
