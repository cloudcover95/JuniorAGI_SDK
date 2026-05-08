import mlx.core as mx
import psutil
import subprocess
from typing import Dict, Any, Tuple
from .base import PhysicalManifold, graceful_degradation
from ..core.config import system_config
from ..core.logger import log

class JuniorHomeManifold(PhysicalManifold):
    """
    Thermodynamic manifold with Adaptive Compute Profiles.
    Scales variational free-energy penalties based on the hardware's actual power source.
    """
    def __init__(self):
        self._name = f"JuniorHome_Thermodynamics_[{system_config.compute_profile}]"
        self._telemetry = {
            "status": "coherent",
            "battery_percent": 100.0,
            "unified_mem_usage": 0.0,
            "cpu_load": 0.0
        }

    @property
    def name(self) -> str:
        return self._name

    def _poll_hardware(self):
        self._telemetry["unified_mem_usage"] = psutil.virtual_memory().percent
        self._telemetry["cpu_load"] = psutil.cpu_percent(interval=None)
        
        if system_config.compute_profile in ["OFF_GRID_48V", "MOBILE_BATTERY"]:
            try:
                batt_info = subprocess.check_output(["pmset", "-g", "batt"]).decode("utf-8")
                if "%" in batt_info:
                    percent_str = batt_info.split("\t")[1].split("%")[0]
                    self._telemetry["battery_percent"] = float(percent_str)
            except Exception:
                pass 

    @graceful_degradation(fallback_tensor_func=lambda: mx.zeros(1), fallback_temp=1.0)
    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        self._poll_hardware()
        
        penalty_scalar = 0.0
        
        # Profile: Grid-Tied (Max Compute, strictly thermal/memory bounded)
        if system_config.compute_profile == "GRID_TIED":
            if self._telemetry["unified_mem_usage"] > 95.0:
                penalty_scalar += 0.3
                self._telemetry["status"] = "memory_pressure_dampening"
            else:
                self._telemetry["status"] = "coherent_grid_power"

        # Profile: Off-Grid or Mobile (Energy constrained)
        else:
            if self._telemetry["battery_percent"] < 20.0:
                penalty_scalar += 0.5
                self._telemetry["status"] = "power_starvation_dampening"
            elif self._telemetry["unified_mem_usage"] > 90.0:
                penalty_scalar += 0.4
                self._telemetry["status"] = "memory_pressure_dampening"
            else:
                self._telemetry["status"] = "coherent_battery_power"
            
        return mx.zeros_like(logits) + penalty_scalar, 1.0

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry