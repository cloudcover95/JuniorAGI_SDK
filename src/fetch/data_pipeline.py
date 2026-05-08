import mlx.core as mx
import pandas as pd
import os
import time

class UnifiedFetchPipeline:
    """
    Vectorized data ingestion pipeline.
    Formats real-world arrays into dense tensors for Synaptic injection.
    """
    def __init__(self, target_dir: str = "assets/data"):
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def ingest_timeseries(self, raw_data: list, source: str) -> mx.array:
        """Converts raw list data to MLX tensors and logs to Parquet."""
        tensor = mx.array(raw_data)
        mx.eval(tensor)
        
        # High-density storage protocol
        df = pd.DataFrame({"source": [source], "timestamp": [time.time()], "tensor_shape": [str(tensor.shape)]})
        df.to_parquet(os.path.join(self.target_dir, f"ingest_{int(time.time())}.parquet"), compression='snappy')
        return tensor
