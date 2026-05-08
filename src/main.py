# /Users/nico/Documents/JuniorCloud/JuniorAGI_SDK/src/main.py

import mlx.core as mx
import threading
from junioragi.reflection_engine import JuniorAGIReflectionNode
from sandbox.llm_bridge import AdaptiveLLMSandbox

class JuniorCloudOrchestrator:
    def __init__(self):
        self.reflection_node = JuniorAGIReflectionNode()
        self.llm_sandbox = AdaptiveLLMSandbox("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
        # Inject the LLM Sandbox into the Reflection Node for UI Modulation
        self.reflection_node.ui_bridge = self.llm_sandbox

    def run(self):
        # Start WebSocket server in a separate thread
        server_thread = threading.Thread(target=self.reflection_node.start_server, daemon=True)
        server_thread.start()
        
        print("JuniorCloud OS: Active. WebSocket listening on 127.0.0.1:8765")
        
        # Infinite telemetry manifold update loop
        while True:
            dummy_data = mx.random.normal((1024, 1024), dtype=mx.float32)
            self.reflection_node.execute_manifold_reduction(dummy_data)

if __name__ == "__main__":
    orchestrator = JuniorCloudOrchestrator()
    orchestrator.run()
