import mlx.core as mx
import logging

logger = logging.getLogger("JuniorAGI.Swarm")

class DistributedMesh:
    """Thunderbolt 5 / IP Mesh for 100B Tensor Parallelism."""
    def __init__(self):
        self.is_distributed, self.world_size, self.rank = False, 1, 0
        try:
            import mlx.distributed as dist
            if dist.is_available():
                self.group = dist.init()
                self.world_size, self.rank = self.group.size(), self.group.rank()
                self.is_distributed = True
                logger.info(f"[+] Swarm Mode Active. Rank: {self.rank}/{self.world_size}")
        except:
            pass

    def all_reduce_tensor(self, tensor: mx.array) -> mx.array:
        if self.is_distributed:
            import mlx.distributed as dist
            return dist.all_sum(tensor, group=self.group)
        return tensor

    def shard_dimension(self, dim: int) -> int:
        if self.is_distributed:
            sharded = dim // self.world_size
            return sharded + (128 - (sharded % 128)) if sharded % 128 != 0 else sharded
        return dim
