import re
import logging
import mlx.core as mx
from typing import Dict, Any, Tuple
from .base import PhysicalManifold, graceful_degradation

logger = logging.getLogger("JuniorLLM.DynamicCompiler")

class DynamicManifoldCompiler:
    """
    AGI-adjacent runtime compiler.
    Parses operational notes containing raw logic rules or Python AST blocks,
    instantiates them as PhysicalManifolds in memory, and binds them to the active SovereignNode.
    """
    def __init__(self, sovereign_node):
        self.node = sovereign_node
        self.pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)

    def compile_and_register(self, raw_note: str):
        """
        Translates raw text logic into executable MLX manifold constraints.
        Extracts code blocks defining a PhysicalManifold subclass.
        """
        matches = self.pattern.findall(raw_note)
        if not matches:
            logger.warning("[-] Dynamic Synthesis failed: No valid Python logic blocks detected in note.")
            return

        for code_block in matches:
            try:
                # Secure local execution dictionary
                exec_globals = {
                    "mx": mx,
                    "PhysicalManifold": PhysicalManifold,
                    "graceful_degradation": graceful_degradation,
                    "Tuple": Tuple,
                    "Dict": Dict,
                    "Any": Any
                }
                exec_locals = {}
                
                # Execute string logic to allocate classes into memory
                exec(code_block, exec_globals, exec_locals)
                
                # Scan local variables for new PhysicalManifold implementations
                registered_count = 0
                for obj_name, obj in exec_locals.items():
                    if isinstance(obj, type) and issubclass(obj, PhysicalManifold) and obj is not PhysicalManifold:
                        # Instantiate and register
                        new_manifold = obj()
                        self.node.register_manifold(new_manifold)
                        registered_count += 1
                        logger.info(f"[+] Dynamic Manifold '{new_manifold.name}' synthesized and bound successfully.")
                
                if registered_count == 0:
                    logger.warning("[-] Code executed, but no valid PhysicalManifold subclasses found.")

            except Exception as e:
                logger.error(f"[-] Catastrophic failure during Dynamic Synthesis: {e}")