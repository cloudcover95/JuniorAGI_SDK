import subprocess
import os
import mlx.core as mx

class HardwareMatrix:
    """
    Hybrid UMA Profiler. Maps both GPU/MPS memory and CPU logical core limits.
    """
    def __init__(self):
        self.chip_info = self._get_profile()
        self.mps_available = mx.metal.is_available()
        self.total_mem = self._get_mem()
        self.cpu_cores = os.cpu_count() or 8

    def _get_profile(self) -> str:
        try: return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        except: return "Apple Silicon"

    def _get_mem(self) -> int:
        try: return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()) // 1024**3
        except: return 8

    def get_specs(self) -> dict:
        return {
            "chip": self.chip_info,
            "mps": self.mps_available,
            "uma_gb": self.total_mem,
            "cpu_cores": self.cpu_cores,
            "bandwidth_tier": "ULTRA" if "ULTRA" in self.chip_info.upper() else "MAX" if "MAX" in self.chip_info.upper() else "STD"
        }
