from infrastructure.hardware_matrix import HardwareMatrix

class InternalAttentionEconomy:
    def __init__(self):
        self.hw = HardwareMatrix()
        self.specs = self.hw.get_specs()
        self.base_budget = 100.0

    def calculate_c2v_ratio(self) -> float:
        multiplier = 1.0
        if self.specs['mps']: multiplier += 0.5
        if self.specs['bandwidth_tier'] == "ULTRA": multiplier += 1.0
        elif self.specs['bandwidth_tier'] == "MAX": multiplier += 0.5
        ratio = (self.base_budget * multiplier) / max(1, (128 / self.specs['uma_gb']))
        return round(ratio, 2)

    def get_telemetry(self) -> dict:
        return {"c2v_ratio": self.calculate_c2v_ratio(), "verified_tier": self.specs['bandwidth_tier']}
