# src/kernel/autonomic_daemon.py
import asyncio
import logging
import mlx.core as mx

logger = logging.getLogger("JuniorAGI.Daemon")

class AutonomicDaemon:
    """
    Continuous Active Inference Engine.
    Minimizes Variational Free Energy during UMA idle cycles.
    """
    def __init__(self, kernel_ref):
        self.kernel = kernel_ref
        self.running = False
        self._task = None

    async def ignite(self):
        self.running = True
        self._task = asyncio.create_task(self._forage_loop())
        logger.info("[+] Autonomic Daemon Ignited. Free Energy Minimization active.")

    async def shutdown(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[-] Autonomic Daemon Halted.")

    async def _forage_loop(self):
        while self.running:
            try:
                # Poll C2V Economy for idle status
                params = self.kernel.orchestrator.get_execution_params()
                
                if params.get('idle', False):
                    await self._execute_active_inference()
                
                # Dynamic sleep based on UMA pressure
                sleep_interval = 1.0 if params.get('idle') else 5.0
                await asyncio.sleep(sleep_interval)
                
            except Exception as e:
                logger.error(f"[!] Daemon Iteration Fault: {e}")
                await asyncio.sleep(5.0)

    async def _execute_active_inference(self):
        """Simulates internal dreaming / residual optimization."""
        # Pull stochastic episodic vector (Simulated for v0.65)
        simulated_memory = mx.random.normal((1, 1024))
        target_prediction = simulated_memory * 0.95  # Simulated decay/entropy
        
        # Execute forward pass through ternary manifold
        # Using self.kernel.layer_0 from agi_kernel
        if hasattr(self.kernel, 'layer_0'):
            y_pred = self.kernel.layer_0(simulated_memory, tau=0.1)
            
            # ABI update via Gamma engine surprise mapping
            self.kernel.orchestrator.gamma_engine.update(y_pred, target_prediction)
            surprise = self.kernel.orchestrator.gamma_engine.gamma_t.item()
            
            if surprise > 0.05:
                # Inject micro-gradient to correct the residual manifold
                self.kernel.plasticity.execute_abi_cycle(
                    self.kernel.layer_0, 
                    simulated_memory, 
                    target_prediction
                )
                logger.debug(f"[*] Active Inference Cycle. Surprise minimized: {surprise:.4f}")
