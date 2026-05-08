# manifolds/temporal_gamma.py
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.gamma_signal import GammaSignalEngine

logger = logging.getLogger("Sovereign.TemporalGamma")

class TemporalGammaManifold(PhysicalManifold):
    """
    Continuous Temporal Interceptor.
    Does not parse text blocks. Instead, intercepts the `get_system_telemetry` payload 
    to append mathematical gradients (d/dt) before Daemon context ingestion.
    """
    def __init__(self, sovereign_node: Any):
        self._name = "Sovereign_Temporal_Gamma"
        self.node = sovereign_node
        self.gamma_engine = GammaSignalEngine()
        self.buffer = ""
        self.buffering = False
        self._latest_gradients = {}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    def process_stream(self, chunk: str) -> str:
        # Passive manifold; does not intercept text tokens
        return chunk

    def intercept_telemetry(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Called by the Sovereign Node prior to returning the telemetry state."""
        self._latest_gradients = self.gamma_engine.record_state_and_compute_gamma(raw_telemetry)
        return self._latest_gradients

    def get_telemetry(self) -> Dict[str, Any]:
        return {"active": True, "latest_gamma_signals": self._latest_gradients}
