# src/infrastructure/distributed_mesh.py
import mlx.core as mx
import logging

logger = logging.getLogger("JuniorAGI.Swarm")

class DistributedMesh:
    """
    Thunderbolt 5 Swarm Interconnect.
    Implements Column/Row Tensor Parallelism for 100B macro-scaling.
    """
    def __init__(self):
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self._init_mesh()

    def _init_mesh(self):
        try:
            import mlx.distributed as dist
            if dist.is_available():
                self.group = dist.init()
                self.world_size = self.group.size()
                self.rank = self.group.rank()
                self.is_distributed = True
                logger.info(f"[+] Swarm Mode Active. Rank: {self.rank}/{self.world_size}")
        except Exception:
            logger.info("[-] Running in Single-Device Sovereign Mode.")

    def all_reduce_tensor(self, tensor: mx.array) -> mx.array:
        """Sum gradients and residuals across the mesh."""
        if self.is_distributed:
            import mlx.distributed as dist
            return dist.all_sum(tensor, group=self.group)
        return tensor

    def shard_dimension(self, dim: int) -> int:
        """Tensor Parallelism: Shatters matrix load across the Swarm."""
        if self.is_distributed:
            return dim // self.world_size
        return dim
