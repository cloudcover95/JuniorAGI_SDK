import ast
import logging
import traceback
from typing import Dict, Any, Type, Optional, Tuple
import mlx.core as mx

from .base import PhysicalManifold, graceful_degradation

logger = logging.getLogger("Sovereign.Sandbox")

class SecurityBoundaryViolation(Exception):
    pass

class ASTSecurityScanner(ast.NodeVisitor):
    def __init__(self):
        self.safe_nodes = {
            ast.Module, ast.ClassDef, ast.FunctionDef, ast.arguments, ast.arg,
            ast.Assign, ast.Store, ast.Name, ast.Load, ast.Return, ast.Expr,
            ast.Call, ast.Attribute, ast.Tuple, ast.List, ast.Constant,
            ast.If, ast.Compare, ast.BinOp, ast.UnaryOp, ast.operator, ast.cmpop,
            ast.Subscript, ast.Slice, ast.Pass, ast.Dict, ast.Set, ast.For, ast.While
        }
        self.allowed_calls = {"len", "range", "float", "int", "str", "bool", "mx", "getattr", "setattr"}

    def generic_visit(self, node):
        if type(node) not in self.safe_nodes:
            raise SecurityBoundaryViolation(f"Illegal AST Node Detected: {type(node).__name__}")
        
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                func_name = node.func.value.id
            
            if func_name not in self.allowed_calls:
                 raise SecurityBoundaryViolation(f"Illegal Function Call Detected: {func_name}")

        super().generic_visit(node)

class ManifoldSandbox:
    """
    Zero-Trust Compiler & Epistemic Simulator.
    Executes AST validation followed by isolated tensor runtime simulation.
    """
    def __init__(self):
        self.allowed_globals = {
            "mx": mx,
            "PhysicalManifold": PhysicalManifold,
            "graceful_degradation": graceful_degradation,
            "Dict": Dict, "Any": Any, "Tuple": Tuple
        }

    def compile_and_simulate(self, source_code: str) -> Tuple[Optional[Type[PhysicalManifold]], Optional[str]]:
        """
        Returns (CompiledClass, None) on success.
        Returns (None, ErrorTraceback) on failure.
        """
        try:
            # 1. Static AST Analysis
            tree = ast.parse(source_code)
            scanner = ASTSecurityScanner()
            scanner.visit(tree)
            
            # 2. Restricted Memory Compilation
            exec_locals = {}
            compiled_code = compile(tree, filename="<dynamic_manifold>", mode="exec")
            exec(compiled_code, self.allowed_globals, exec_locals)
            
            manifold_class = None
            for obj_name, obj in exec_locals.items():
                if isinstance(obj, type) and issubclass(obj, PhysicalManifold) and obj is not PhysicalManifold:
                    manifold_class = obj
                    break
            
            if not manifold_class:
                return None, "AST Valid, but no PhysicalManifold subclass defined."

            # 3. Epistemic Runtime Simulation (Dream State)
            instance = manifold_class()
            
            # Simulate vocabulary dimension MLX tensor (e.g., 151936 vocab size)
            dummy_input_ids = mx.array([[1, 2, 3]])
            dummy_logits = mx.random.normal((1, 3, 151936))
            
            # Execute logic against dummy tensors to catch shape/math failures
            penalty, scalar = instance.compute_surprise(dummy_input_ids, dummy_logits)
            
            # Force MLX evaluation to catch deferred compute graph errors
            mx.eval(penalty)
            
            logger.info(f"[+] Epistemic Simulation Passed: {instance.name}")
            return manifold_class, None

        except SecurityBoundaryViolation as e:
            return None, f"SecurityBoundaryViolation: {str(e)}"
        except Exception as e:
            error_trace = traceback.format_exc()
            return None, f"RuntimeSimulationCollapse:\n{error_trace}"
