import mlx.core as mx
from infrastructure.hardware_matrix import HardwareMatrix

class MPSEngine:
    def __init__(self):
        self.hw = HardwareMatrix()
        if not self.hw.mps_available:
            raise RuntimeError("JuniorAGI requires MPS/Metal on Apple Silicon.")

    def evaluate_tensor(self, x: mx.array) -> mx.array:
        mx.eval(x)
        return x

    def get_telemetry(self) -> dict:
        return {"vram_mb": mx.metal.get_active_memory() / 1024**2}
