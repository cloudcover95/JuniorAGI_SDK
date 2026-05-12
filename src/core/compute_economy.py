# src/core/compute_economy.py
import psutil
import mlx.core as mx
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    """
    Thermodynamic Estimator & C2V Router.
    Maps physical hardware TDP to calculate true Joules Per Inference (JPI).
    """
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()
        
        # Apple Silicon Empirical TDP Brackets (Watts)
        if self.specs['bandwidth_tier'] == "ULTRA": self.tdp_idle, self.tdp_max = 12.0, 100.0
        elif self.specs['bandwidth_tier'] == "MAX": self.tdp_idle, self.tdp_max = 8.0, 60.0
        elif self.specs['bandwidth_tier'] == "CPU_ONLY": self.tdp_idle, self.tdp_max = 15.0, 150.0
        else: self.tdp_idle, self.tdp_max = 4.0, 30.0

    def _get_vram_utilization(self) -> float:
        if self.specs['mps']:
            mem_func = getattr(mx.metal, 'get_active_memory', None) or getattr(mx, 'get_active_memory')
            return (mem_func() / 1024**3) / self.specs['uma_gb']
        return psutil.virtual_memory().percent / 100.0

    def calculate_jpi(self, latency_seconds: float) -> float:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        vram_util = self._get_vram_utilization()
        
        # Hardware-weighted utilization (Apple Silicon leans heavily on GPU/ANE)
        util_factor = (cpu_load * 0.3) + (min(1.0, vram_util) * 0.7)
        est_watts = self.tdp_idle + ((self.tdp_max - self.tdp_idle) * util_factor)
        
        # E (Joules) = P (Watts) * t (Seconds)
        return round(est_watts * latency_seconds, 4)

    def get_c2v_metrics(self) -> dict:
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        vram_util = self._get_vram_utilization()
        
        cpu_headroom = max(0.01, 1.0 - cpu_load)
        uma_headroom = max(0.01, 1.0 - vram_util)
        
        # Synthetic thermal proxy based on sustained UMA memory controller saturation
        thermal_pressure = min(1.0, (vram_util * 0.8) + (cpu_load * 0.2))
        
        # Power Budget drops dynamically as thermal pressure compounds
        power_budget = (cpu_headroom * 0.3) + (uma_headroom * 0.5) + ((1.0 - thermal_pressure) * 0.2)
        
        return {
            "power_budget": round(power_budget, 4),
            "thermal_pressure": round(thermal_pressure, 4),
            "uma_utilization": round(vram_util, 4)
        }
