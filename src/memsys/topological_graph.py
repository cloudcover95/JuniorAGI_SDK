import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger("Sovereign.Graph")

class TopologicalGraph:
    """
    Zero-Dependency Edge-Native Relational Knowledge Graph.
    Maps Betti-1 topological loops and multi-hop deductive relationships.
    """
    def __init__(self, db_path: str = "~/.juniorllm/relational_mesh.json"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.adjacency_list: Dict[str, Dict[str, List[str]]] = {}
        self._load_graph()

    def _load_graph(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.adjacency_list = json.load(f)
                logger.info(f"[+] Topological Graph loaded. Nodes: {len(self.adjacency_list)}")
            except Exception as e:
                logger.error(f"[-] Graph compilation failure: {e}")

    def flush_to_disk(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.adjacency_list, f, indent=2)
        except Exception as e:
            logger.error(f"[-] Graph disk flush failure: {e}")

    def insert_edge(self, subject: str, predicate: str, obj: str):
        subject = subject.upper().strip()
        predicate = predicate.upper().strip()
        obj = obj.upper().strip()

        if subject not in self.adjacency_list:
            self.adjacency_list[subject] = {}
        if predicate not in self.adjacency_list[subject]:
            self.adjacency_list[subject][predicate] = []
            
        if obj not in self.adjacency_list[subject][predicate]:
            self.adjacency_list[subject][predicate].append(obj)
            self.flush_to_disk()
            logger.info(f"[*] Betti-1 Edge Formed: {subject} -[{predicate}]-> {obj}")

    def extract_subgraph(self, target_nodes: List[str], depth: int = 2) -> str:
        """
        Executes multi-hop traversal to pull related structural logic.
        Returns a formatted string of relationships for LLM context injection.
        """
        if not self.adjacency_list:
            return ""

        visited = set()
        queue = [(node.upper(), 0) for node in target_nodes if node.upper() in self.adjacency_list]
        extracted_edges = []

        while queue:
            current_node, current_depth = queue.pop(0)
            if current_node in visited or current_depth >= depth:
                continue
                
            visited.add(current_node)
            
            if current_node in self.adjacency_list:
                for predicate, objects in self.adjacency_list[current_node].items():
                    for obj in objects:
                        extracted_edges.append(f"({current_node}) -[{predicate}]-> ({obj})")
                        if obj not in visited:
                            queue.append((obj, current_depth + 1))

        if not extracted_edges:
            return ""
            
        return "\n".join(set(extracted_edges))

    def get_telemetry(self) -> Dict[str, Any]:
        return {"total_nodes": len(self.adjacency_list)}
