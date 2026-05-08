# manifolds/alignment.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from core.value_tensor import SVDValueManifold

logger = logging.getLogger("Sovereign.Alignment")

class AlignmentManifold(PhysicalManifold):
    """
    Teleological Interceptor.
    Evaluates [[PLAN: ...]] blocks. Aborts generation if the mathematical alignment 
    with the SVD Value Manifold falls below thermodynamic/enterprise thresholds.
    """
    def __init__(self, sovereign_node: Any):
        self._name = "Sovereign_Teleological_Alignment"
        self.node = sovereign_node
        self.value_engine = SVDValueManifold(k_components=16)
        
        self.plan_pattern = re.compile(r"\[\[PLAN:\s*(.*?)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"plans_evaluated": 0, "pruned_by_misalignment": 0}
        self.alignment_threshold = 0.45 

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    def _bootstrap_value_manifold(self):
        if self.node.synapse.table is not None and not self.value_engine.is_calibrated:
            df = self.node.synapse.table.to_pandas()
            high_value_df = df[df['plasticity_score'] > 1.2]
            if len(high_value_df) > 10:
                vectors = mx.array(high_value_df['vector'].tolist())
                self.value_engine.recompute_manifold(vectors)

    async def _evaluate_teleology(self, plan_text: str):
        self._bootstrap_value_manifold()
        
        if not self.value_engine.is_calibrated:
            return

        try:
            tokens = mx.array([self.node.tokenizer.encode(plan_text)])
            if hasattr(self.node.model, "model") and hasattr(self.node.model.model, "embed_tokens"):
                embeds = self.node.model.model.embed_tokens(tokens)
                p_vector = mx.mean(embeds, axis=1)[0]
                
                score = self.value_engine.compute_alignment_score(p_vector)
                
                if score < self.alignment_threshold:
                    self._telemetry["pruned_by_misalignment"] += 1
                    logger.critical(f"[!] ORTHOGONAL TELEOLOGY DETECTED. Plan alignment ({score:.2f}) below threshold ({self.alignment_threshold}).")
                    
                    # Abort ongoing generation
                    self.node.trigger_reflex_preemption()
                    
                    error_payload = f"VALUE_MISALIGNMENT:\nThe preceding [[PLAN]] yielded an alignment scalar of {score:.2f}. " \
                                    f"It fails to optimize JuniorCloud LLC operations, financial nodes, or structural kinematics. " \
                                    f"Re-evaluate priorities and generate an orthogonal correction."
                    await self.node.internal_feedback.put(error_payload)
                else:
                    logger.info(f"[*] Teleology Aligned. SVD Scalar: {score:.2f}")
                    
        except Exception as e:
            logger.error(f"[-] Alignment verification collapse: {e}")

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.plan_pattern.search(self.buffer)
        if match:
            plan_text = match.group(1).strip()
            self._telemetry["plans_evaluated"] += 1
            
            # Asynchronously evaluate alignment to prevent generation blocking
            asyncio.create_task(self._evaluate_teleology(plan_text))
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 2048 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry
