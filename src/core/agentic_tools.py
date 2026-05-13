import os, subprocess, logging, time
from typing import Dict, Any

logger = logging.getLogger("JuniorAGI.Tools")

class SovereignAgentTools:
    def __init__(self, sandbox_dir: str = "assets/sandbox"):
        self.sandbox = os.path.abspath(sandbox_dir)
        os.makedirs(self.sandbox, exist_ok=True)
        self.registered_tools = {
            "read_file": self._read, "write_file": self._write,
            "list_dir": self._list, "system_time": self._time
        }

    def _bound(self, path: str) -> str:
        t = os.path.abspath(os.path.join(self.sandbox, path))
        if not t.startswith(self.sandbox): raise PermissionError("Sandbox Violation")
        return t

    def _read(self, p: Dict) -> str:
        pth = self._bound(p.get("filename", ""))
        return open(pth, 'r').read() if os.path.exists(pth) else "Err: Not found."

    def _write(self, p: Dict) -> str:
        pth = self._bound(p.get("filename", ""))
        with open(pth, 'w') as f: f.write(p.get("content", ""))
        return f"Wrote {pth}"

    def _list(self, p: Dict) -> str:
        pth = self._bound(p.get("directory", ""))
        return "\n".join(os.listdir(pth)) if os.path.exists(pth) else "Err: Not found."

    def _time(self, p: Dict) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S %Z")

    def execute(self, name: str, params: Dict) -> str:
        if name not in self.registered_tools: return f"Err: Unknown tool {name}"
        logger.info(f"[*] Tool Invoked: {name}")
        return self.registered_tools[name](params)
