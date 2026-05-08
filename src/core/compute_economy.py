import psutil
from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    """
    C2V Allocator with CPU Headroom Detection.
    Determines if tasks should execute via GPU Metal shaders or CPU UMA offloading.
    """
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()
        self.base_budget = 100.0

    def calculate_c2v_ratio(self) -> dict:
        # GPU Multiplier
        gpu_mult = 1.5 if self.specs['mps'] else 1.0
        if self.specs['bandwidth_tier'] == "ULTRA": gpu_mult += 1.0
        elif self.specs['bandwidth_tier'] == "MAX": gpu_mult += 0.5
        
        # CPU Headroom (Inverse of current load)
        cpu_load = psutil.cpu_percent(interval=None) / 100.0
        cpu_headroom = max(0.1, 1.0 - cpu_load)
        cpu_mult = (self.specs['cpu_cores'] / 8.0) * cpu_headroom

        mem_pressure = max(1, (128 / self.specs['uma_gb']))
        
        return {
            "gpu_c2v": round((self.base_budget * gpu_mult) / mem_pressure, 2),
            "cpu_c2v": round((self.base_budget * cpu_mult) / mem_pressure, 2),
            "cpu_headroom_pct": round(cpu_headroom * 100, 1)
        }

    def get_telemetry(self) -> dict:
        ratios = self.calculate_c2v_ratio()
        return {
            "c2v_ratios": ratios,
            "verified_tier": self.specs['bandwidth_tier'],
            "hybrid_ready": True
        }
