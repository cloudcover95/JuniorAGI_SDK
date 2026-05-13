import psutil, time
import mlx.core as mx
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()
        t = self.specs['tier']
        self.tdp_idle, self.tdp_max = (12., 100.) if t == "ULTRA" else (8., 60.) if t == "MAX" else (15., 150.) if t == "CPU_ONLY" else (4., 30.)
        self.baseline_load = psutil.cpu_percent(interval=0.1) / 100.0

    def _get_vram_util(self) -> float:
        # Modern MLX API (Deprecation fixed)
        if self.specs['mps'] and hasattr(mx, 'get_active_memory'):
            return (mx.get_active_memory() / 1024**3) / self.specs['uma_gb']
        return psutil.virtual_memory().percent / 100.0

    def get_c2v_metrics(self) -> dict:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        vram_util = self._get_vram_util()
        
        cpu_headroom = max(0.01, 1.0 - cpu_load)
        uma_headroom = max(0.01, 1.0 - vram_util)
        thermal_pressure = min(1.0, (vram_util * 0.8) + (cpu_load * 0.2))
        power_budget = (cpu_headroom * 0.3) + (uma_headroom * 0.5) + ((1.0 - thermal_pressure) * 0.2)
        
        return {
            "power_budget": round(power_budget, 4),
            "thermal_pressure": round(thermal_pressure, 4),
            "background_load": round(self.baseline_load, 4),
            "uma_utilization": round(vram_util, 4)
        }

    def calculate_jpi(self, latency: float) -> float:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        vram_util = self._get_vram_util()
        util_factor = (cpu_load * 0.3) + (min(1.0, vram_util) * 0.7)
        watts = self.tdp_idle + ((self.tdp_max - self.tdp_idle) * util_factor)
        return round(watts * latency, 4)
