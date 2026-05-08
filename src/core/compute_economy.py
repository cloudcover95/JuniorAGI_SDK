# src/core/compute_economy.py
import psutil
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()

    def get_c2v_metrics(self) -> dict:
        """
        Derives Power Budget and C2V Ratio.
        power_budget determines dynamic bit-width quantization in the BitNet layers.
        """
        # Inverse mapping of system pressure
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        cpu_headroom = max(0.1, 1.0 - cpu_load)
        
        import mlx.core as mx
        mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
        vram_used = (mem_func() / 1024**3)
        uma_headroom = max(0.1, 1.0 - (vram_used / self.specs['uma_gb']))
        
        # Power budget interpolates between 0.0 (Throttle) and 1.0 (Full Power)
        power_budget = (cpu_headroom * 0.4) + (uma_headroom * 0.6)
        
        return {
            "power_budget": round(power_budget, 4),
            "cpu_headroom": round(cpu_headroom, 4),
            "uma_headroom": round(uma_headroom, 4)
        }
