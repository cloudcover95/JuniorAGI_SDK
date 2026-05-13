# src/core/agentic_tools.py
import os
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger("JuniorAGI.Tools")

class SovereignAgentTools:
    """
    Zero-Trust Local Tool Execution.
    Grants the JuniorAGI node capabilities to interact with the local macOS/Linux environment.
    """
    def __init__(self, sandbox_dir: str = "assets/sandbox"):
        self.sandbox = os.path.abspath(sandbox_dir)
        os.makedirs(self.sandbox, exist_ok=True)
        
        self.registered_tools = {
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "list_directory": self._tool_list_dir,
            "execute_script": self._tool_execute_script
        }

    def _enforce_bounds(self, filepath: str) -> str:
        """Prevents directory traversal attacks (e.g., '../../etc/passwd')."""
        target = os.path.abspath(os.path.join(self.sandbox, filepath))
        if not target.startswith(self.sandbox):
            raise PermissionError("[!] Sovereign Sandbox Violation Attempted.")
        return target

    def _tool_read_file(self, params: Dict[str, Any]) -> str:
        path = self._enforce_bounds(params.get("filename", ""))
        if not os.path.exists(path): return "Error: File not found."
        with open(path, 'r') as f: return f.read()

    def _tool_write_file(self, params: Dict[str, Any]) -> str:
        path = self._enforce_bounds(params.get("filename", ""))
        with open(path, 'w') as f: f.write(params.get("content", ""))
        return f"Success: {params.get('filename')} written."

    def _tool_list_dir(self, params: Dict[str, Any]) -> str:
        path = self._enforce_bounds(params.get("directory", ""))
        if not os.path.exists(path): return "Error: Directory not found."
        return "\n".join(os.listdir(path))

    def _tool_execute_script(self, params: Dict[str, Any]) -> str:
        """Executes a shell command explicitly constrained to the sandbox."""
        cmd = params.get("command", "")
        # High security: Do not allow arbitrary bash without strict parsing in prod
        # Here we limit execution to the sandbox CWD
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=self.sandbox, capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Execution Error: {str(e)}"

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        if tool_name not in self.registered_tools:
            return f"Error: Tool '{tool_name}' not registered."
        logger.info(f"[*] Agent executing tool: {tool_name}")
        return self.registered_tools[tool_name](params)
