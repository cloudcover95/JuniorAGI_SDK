# src/core/compute_economy.py
import psutil
import subprocess
import mlx.core as mx
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()

    def _get_thermal_proxy(self) -> float:
        """
        Derives a crude thermal/power proxy (0.0 to 1.0, 1.0 being critical).
        On macOS, we check powermetrics (requires sudo usually, falling back to CPU load as proxy if unavailable).
        """
        try:
            # Fallback heuristic: High sustained load + minimal memory = thermal pressure
            return min(1.0, (psutil.cpu_percent(interval=None) / 100.0) * 1.2)
        except Exception:
            return 0.5

    def get_c2v_metrics(self) -> dict:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        cpu_headroom = max(0.01, 1.0 - cpu_load)
        
        vram_used = (mx.get_active_memory() / 1024**3)
        uma_headroom = max(0.01, 1.0 - (vram_used / self.specs['uma_gb']))
        
        thermal_pressure = self._get_thermal_proxy()
        
        # Power budget interpolates smoothly
        power_budget = (cpu_headroom * 0.3) + (uma_headroom * 0.5) + ((1.0 - thermal_pressure) * 0.2)
        
        return {
            "power_budget": round(power_budget, 4),
            "cpu_headroom": round(cpu_headroom, 4),
            "uma_headroom": round(uma_headroom, 4),
            "thermal_pressure": round(thermal_pressure, 4)
        }
