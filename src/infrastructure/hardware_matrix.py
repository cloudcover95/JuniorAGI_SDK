import subprocess
import mlx.core as mx

class HardwareMatrix:
    def __init__(self):
        self.chip_info = self._get_profile()
        self.mps_available = mx.metal.is_available()
        self.total_mem = self._get_mem()

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
            "bandwidth_tier": "ULTRA" if "ULTRA" in self.chip_info.upper() else "MAX" if "MAX" in self.chip_info.upper() else "STD"
        }
