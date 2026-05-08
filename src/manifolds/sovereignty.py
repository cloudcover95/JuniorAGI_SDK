import mlx.core as mx
from typing import Dict, Any, List, Tuple
from .base import PhysicalManifold, graceful_degradation

class SovereigntyManifold(PhysicalManifold):
    """
    O(1) overhead Sovereignty Identity Manifold.
    Pre-compiles rigid masks to execute immediate tensor addition, 
    bypassing dynamic index lookups in the active inference stream.
    """
    def __init__(self, tokenizer: Any, vocab_size: int = 151936):
        self._name = "Sovereignty_Identity"
        self.tokenizer = tokenizer
        
        forbidden_strings = ["As an AI", "I cannot", "I am a large language model", "I apologize"]
        forbidden_tokens: List[int] = []
        
        for phrase in forbidden_strings:
            tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
            if tokens:
                forbidden_tokens.extend(tokens)
                
        forbidden_tokens = list(set(forbidden_tokens))
        self.forbidden_tokens_set = set(forbidden_tokens)
        
        # Pre-allocate and evaluate static topological mask
        self.penalty_mask = mx.zeros(vocab_size)
        if forbidden_tokens:
            indices = mx.array(forbidden_tokens)
            self.penalty_mask[indices] = 100.0
            mx.eval(self.penalty_mask)
            
        self._telemetry = {"status": "sovereign_lock_active", "interventions": 0}

    @property
    def name(self) -> str:
        return self._name

    @graceful_degradation(fallback_tensor_func=lambda: mx.array(0.0), fallback_temp=1.0)
    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        if not self.forbidden_tokens_set:
            return mx.zeros_like(logits), 1.0

        # Off-cycle telemetry update (zero computational drag on forward pass)
        top_token = mx.argmax(logits).item()
        if top_token in self.forbidden_tokens_set:
            self._telemetry["interventions"] += 1
            
        # Zero-overhead vectorized projection
        return self.penalty_mask, 1.0

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry