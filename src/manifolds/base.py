import logging
import mlx.core as mx
from typing import Protocol, Dict, Any, Callable, Tuple
from functools import wraps

logger = logging.getLogger("JuniorLLM.Manifold")

class PhysicalManifold(Protocol):
    """
    Protocol defining the structural interface for physical grounding manifolds.
    Enforces active-inference constraints on the LLM's probability space.
    """
    @property
    def name(self) -> str:
        ...

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        """
        Calculates variational free energy penalty and dynamic temperature scalar.
        Returns:
            Tuple[mx.array, float]: (Penalty tensor of shape vocab_size to subtract from logits, 
                                     Temperature scalar multiplier)
        """
        ...
        
    def get_telemetry(self) -> Dict[str, Any]:
        """Returns physical state invariants for UI dashboard routing."""
        ...

def graceful_degradation(fallback_tensor_func: Callable[[], mx.array], fallback_temp: float = 1.0):
    """
    Philosophy: "If one section fails, it becomes part of the user experience."
    Wraps manifold computations to prevent node collapse, logging telemetry on failure.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"[!] Manifold Drift in {getattr(self, 'name', 'Unknown')}: {e}. Degrading gracefully.")
                if hasattr(self, '_telemetry'):
                    self._telemetry['status'] = f"degraded: {str(e)}"
                return fallback_tensor_func(), fallback_temp
        return wrapper
    return decorator