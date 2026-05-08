import mlx.core as mx
from typing import Dict, Any, List, Tuple
from .base import PhysicalManifold, graceful_degradation

class SensorimotorManifold(PhysicalManifold):
    """
    O(1) Sensorimotor Grounding Manifold.
    Applies structural `betti_0` void-proofing via pure vectorized algebra.
    """
    def __init__(self, tokenizer: Any, vocab_size: int = 151936):
        self._name = "JuniorFetch_Sensorimotor"
        self.tokenizer = tokenizer
        
        spatial_strings = ["forward", "advance", "move", "walk", "proceed", "step"]
        spatial_tokens: List[int] = []
        
        for phrase in spatial_strings:
            tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
            if tokens:
                spatial_tokens.extend(tokens)
                
        self.spatial_tokens_set = set(spatial_tokens)
        
        # Pre-allocate rigid array
        self.spatial_penalty_mask = mx.zeros(vocab_size)
        if spatial_tokens:
            indices = mx.array(spatial_tokens)
            self.spatial_penalty_mask[indices] = 15.0
            mx.eval(self.spatial_penalty_mask)
        
        self._telemetry = {
            "status": "coherent",
            "betti_0_obstacles": 1.0, 
            "betti_1_voids": 0.0,
            "spatial_interventions": 0
        }

    @property
    def name(self) -> str:
        return self._name

    def _ingest_lidar_topology(self):
        """Orthogonal CPU hook to update TDA Betti numbers."""
        pass

    @graceful_degradation(fallback_tensor_func=lambda: mx.zeros(1), fallback_temp=1.0)
    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        self._ingest_lidar_topology()
        
        # Vectorized multiplier: (1.0 if obstacle, 0.0 if clear)
        multiplier = float(self._telemetry["betti_0_obstacles"] > 0)
        
        # Zero-overhead projection into logit space
        penalty = self.spatial_penalty_mask * multiplier

        if multiplier > 0:
            self._telemetry["status"] = "obstacle_detected_dampening"
            top_token = mx.argmax(logits).item()
            if top_token in self.spatial_tokens_set:
                self._telemetry["spatial_interventions"] += 1
        else:
            self._telemetry["status"] = "path_clear"
            
        return penalty, 1.0

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry