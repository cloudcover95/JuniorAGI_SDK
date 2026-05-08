# src/core/compute_economy.py
import psutil
import time
import mlx.core as mx
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()
        
        # Apple Silicon Empirical TDP Approximations (Watts)
        if self.specs['bandwidth_tier'] == "ULTRA":
            self.tdp_idle, self.tdp_max = 12.0, 100.0
        elif self.specs['bandwidth_tier'] == "MAX":
            self.tdp_idle, self.tdp_max = 8.0, 60.0
        else:
            self.tdp_idle, self.tdp_max = 4.0, 30.0

    def calculate_jpi(self, latency_seconds: float) -> float:
        """
        Calculates Joules Per Inference (JPI).
        $E (Joules) = \bar{P} (Watts) \times \Delta t (Seconds)$
        """
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        vram_util = (mx.get_active_memory() / 1024**3) / self.specs['uma_gb']
        
        # Weight GPU heavier on M-Series due to large Neural Engine/GPU clusters
        utilization_factor = (cpu_load * 0.3) + (min(1.0, vram_util) * 0.7)
        estimated_watts = self.tdp_idle + ((self.tdp_max - self.tdp_idle) * utilization_factor)
        
        return round(estimated_watts * latency_seconds, 4)

    def get_c2v_metrics(self) -> dict:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        cpu_headroom = max(0.01, 1.0 - cpu_load)
        vram_util = (mx.get_active_memory() / 1024**3) / self.specs['uma_gb']
        uma_headroom = max(0.01, 1.0 - vram_util)
        
        # Dynamic Thermal Proxy (Utilizing sustained VRAM saturation as heat generator)
        thermal_pressure = min(1.0, (vram_util * 0.8) + (cpu_load * 0.2))
        
        # Power budget drops as thermal pressure rises
        power_budget = (cpu_headroom * 0.3) + (uma_headroom * 0.5) + ((1.0 - thermal_pressure) * 0.2)
        
        return {
            "power_budget": round(power_budget, 4),
            "thermal_pressure": round(thermal_pressure, 4),
            "uma_utilization": round(vram_util, 4)
        }
