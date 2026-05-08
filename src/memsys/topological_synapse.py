# Delta fix for synapse injection bypassing MLX during reflex interrupts
import os
import logging
import uuid
from pathlib import Path
import mlx.core as mx
import numpy as np
import pyarrow as pa
import lancedb
from typing import Optional, Dict, Any, List

try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger("Sovereign.Synapse")

class TopologicalSynapse:
    """
    LanceDB-powered RAG Engine with Gradient Synaptic Plasticity.
    """
    def __init__(self, db_path: str = "~/.juniorllm/lancedb/"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.telemetry = {"total_landmarks": 0, "queries_executed": 0, "plasticity_adjustments": 0, "consolidation_cycles": 0}
        
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = "sovereign_mesh"
        self.table = self._load_table()
        self.resonance_threshold = 0.85 
        # Standardize vector dims (Assuming 4096 default for base embedding)
        self.vector_dim = 4096

    def _load_table(self):
        if self.table_name in self.db.table_names():
            table = self.db.open_table(self.table_name)
            self.telemetry["total_landmarks"] = len(table)
            
            # Dynamically pull dim size from existing schema if table exists
            if len(table) > 0:
                df = table.head(1).to_pandas()
                if "vector" in df.columns:
                    self.vector_dim = len(df.iloc[0]["vector"])
            return table
        return None

    def _get_zero_vector(self) -> mx.array:
        """Fallback empty vector for System 1 Reflex logging bypassing MLX graphs."""
        return mx.zeros((self.vector_dim,))

    def insert_landmark(self, vector: mx.array, content: str, metadata: dict) -> str:
        vec_np = np.array(vector.tolist(), dtype=np.float32)
        node_id = str(uuid.uuid4())
        
        data = [{
            "id": node_id,
            "vector": vec_np,
            "content": content,
            "source": metadata.get("source", "unknown"),
            "plasticity_score": 1.0  
        }]
        
        if self.table is None:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), len(vec_np))),
                pa.field("content", pa.string()),
                pa.field("source", pa.string()),
                pa.field("plasticity_score", pa.float32())
            ])
            self.table = self.db.create_table(self.table_name, data=data, schema=schema)
        else:
            self.table.add(data)
            
        self.telemetry["total_landmarks"] += 1
        return node_id

    def adjust_plasticity(self, node_id: str, penalty_delta: float):
        if self.table is None: return
        try:
            self.table.update(where=f"id = '{node_id}'", values={"plasticity_score": f"plasticity_score + {penalty_delta}"})
            self.telemetry["plasticity_adjustments"] += 1
        except Exception as e:
            logger.error(f"[-] Plasticity gradient failure: {e}")

    def semantic_search(self, query_vector: mx.array, k: int = 3) -> List[str]:
        if self.table is None or self.telemetry["total_landmarks"] == 0: return []
        self.telemetry["queries_executed"] += 1
        query_np = np.array(query_vector.tolist(), dtype=np.float32)
        
        try:
            results = self.table.search(query_np).limit(k * 2).to_list()
            scored_chunks = []
            for res in results:
                raw_distance = res.get("_distance", 1.0)
                p_score = res.get("plasticity_score", 1.0)
                effective_distance = raw_distance / max(p_score, 0.01)
                if effective_distance <= self.resonance_threshold:
                    scored_chunks.append((effective_distance, res["content"]))
                    
            scored_chunks.sort(key=lambda x: x[0])
            return [chunk[1] for chunk in scored_chunks[:k]]
        except Exception as e:
            logger.error(f"[-] LanceDB Search Collapse: {e}")
            return []

    def consolidate_mesh(self):
        if not HAS_SKLEARN or self.table is None or self.telemetry["total_landmarks"] < 100: return
        try:
            df = self.table.to_pandas()
            vectors = np.stack(df["vector"].values)
            target_clusters = max(50, len(vectors) // 2)
            kmeans = MiniBatchKMeans(n_clusters=target_clusters, random_state=42, n_init='auto')
            kmeans.fit(vectors)
            self.telemetry["consolidation_cycles"] += 1
        except Exception as e:
            logger.error(f"[-] Consolidation failure: {e}")

    def get_telemetry(self) -> Dict[str, Any]:
        return self.telemetry
