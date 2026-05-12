# src/infrastructure/hardware_matrix.py
import subprocess
import os
import platform
import mlx.core as mx

class HardwareMatrix:
    def __init__(self):
        self.os_type = platform.system()
        self.chip_info = self._get_profile()
        self.mps_available = mx.metal.is_available() if hasattr(mx, 'metal') else False
        self.total_mem = self._get_mem()
        self.cpu_cores = os.cpu_count() or 8

    def _get_profile(self) -> str:
        if self.os_type == "Darwin":
            try: return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            except: return "Apple Silicon"
        return platform.processor() or "Generic CPU"

    def _get_mem(self) -> int:
        if self.os_type == "Darwin":
            try: return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()) // 1024**3
            except: return 8
        try:
            import psutil
            return psutil.virtual_memory().total // 1024**3
        except: return 16

    def get_specs(self) -> dict:
        tier = "STD"
        if "ULTRA" in self.chip_info.upper(): tier = "ULTRA"
        elif "MAX" in self.chip_info.upper(): tier = "MAX"
        elif not self.mps_available: tier = "CPU_ONLY"
        
        return {
            "chip": self.chip_info,
            "mps": self.mps_available,
            "uma_gb": self.total_mem,
            "cpu_cores": self.cpu_cores,
            "bandwidth_tier": tier
        }
