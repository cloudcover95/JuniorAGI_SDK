# manifolds/juniorquant.py
import re
import logging
import asyncio
from typing import Dict, Any, Tuple
import mlx.core as mx

from .base import PhysicalManifold
from fetch.quant_feed import FinancialKinematicsIntake
from core.gamma_signal import GammaSignalEngine

logger = logging.getLogger("Sovereign.JuniorQuant")

class JuniorQuantManifold(PhysicalManifold):
    """
    Economic Interceptor (`JuniorAGI_SDK` bridge).
    Routes HFFM and Web3 data through the Gamma Signal Engine to calculate market velocity,
    treating financial liquidity exactly like physical thermodynamics.
    """
    def __init__(self, sovereign_node: Any, gamma_engine: GammaSignalEngine):
        self._name = "Sovereign_Financial_Kinematics"
        self.node = sovereign_node
        self.feed = FinancialKinematicsIntake()
        self.gamma = gamma_engine # Shared reference to maintain continuous momentum
        
        self.quant_pattern = re.compile(r"\[\[QUANT_ANALYZE:\s*(.*?)\]\]", re.DOTALL)
        self.buffer = ""
        self.buffering = False
        self._telemetry = {"matrices_processed": 0, "latest_asset": "None"}

    @property
    def name(self) -> str:
        return self._name

    def compute_surprise(self, input_ids: mx.array, logits: mx.array) -> Tuple[mx.array, float]:
        return mx.zeros_like(logits), 1.0

    async def _execute_quant_analysis(self, target: str):
        logger.warning(f"[!] Financial Kinematics Initiated: Mapping {target} topology.")
        
        if target.upper() == "WEB3_BASE":
            result = await self.feed.fetch_web3_topology()
            asset_key = "ETH_GAS"
        else:
            result = await self.feed.fetch_market_matrix(target)
            asset_key = target.upper()

        if result["status"] == "SUCCESS":
            self._telemetry["matrices_processed"] += 1
            self._telemetry["latest_asset"] = asset_key
            
            metrics = result["metrics"]
            
            # Inject into Gamma Engine to track momentum (velocity/acceleration)
            # Prefixing to isolate from thermodynamic keys
            gamma_injection = {f"JuniorQuant_{asset_key}.{k}": v for k, v in metrics.items()}
            
            # Note: In a production run, this needs to tick continuously. 
            # For this intercept, we record the state and pull the instantaneous gradient.
            gradients = self.gamma.record_state_and_compute_gamma(gamma_injection)
            
            # Extract specific gradients for the payload
            v_signal = 0.0
            for k, v in gradients.items():
                if "latest_close" in k or "base_fee" in k:
                    v_signal = v.get("velocity_per_sec", 0.0)
            
            payload = f"[FINANCIAL_KINEMATICS: {asset_key}]\nStatic Metrics: {metrics}\nGamma Velocity (d/dt): {v_signal:.4f}"
        else:
            payload = f"[QUANT_ERROR: {result['message']}]"
            
        await self.node.hive_feedback.put({"agent": "JUNIOR_QUANT_NODE", "result": payload})

    def process_stream(self, chunk: str) -> str:
        if not self.buffering:
            if "[" in chunk:
                self.buffering = True
                self.buffer += chunk
                return ""
            return chunk

        self.buffer += chunk
        
        match = self.quant_pattern.search(self.buffer)
        if match:
            target = match.group(1).strip()
            asyncio.create_task(self._execute_quant_analysis(target))
            
            remainder = self.buffer[match.end():]
            self.buffer = ""
            self.buffering = False
            return remainder

        if len(self.buffer) > 1024 or "\n" in chunk:
            dump = self.buffer
            self.buffer = ""
            self.buffering = False
            return dump

        return ""

    def get_telemetry(self) -> Dict[str, Any]:
        return self._telemetry
