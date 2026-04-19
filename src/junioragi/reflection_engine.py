# /Users/nico/Documents/JuniorCloud/JuniorAGI_SDK/src/junioragi/reflection_engine.py

import mlx.core as mx
import numpy as np
import json
import asyncio
import websockets
import logging

class JuniorAGIReflectionNode:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.tensor_state = mx.zeros((1024, 1024), dtype=mx.float32)
        self.ui_bridge = None # Injected by main.py

    def execute_manifold_reduction(self, telemetry_data: mx.array) -> mx.array:
        """MLX CPU-streamed SVD implementation ($A = U \Sigma V^T$)."""
        try:
            U, S, Vt = mx.linalg.svd(telemetry_data, stream=mx.cpu)
            self.tensor_state = mx.matmul(U, mx.diag(S))
        except:
            # Numpy Fallback for Cross-Platform Integrity
            A_np = np.array(telemetry_data.tolist(), dtype=np.float32)
            U, S, Vt = np.linalg.svd(A_np, full_matrices=False)
            self.tensor_state = mx.array(np.dot(U, np.diag(S)).tolist())
        return self.tensor_state

    async def _handle_connection(self, websocket, path):
        """Asynchronous modulation loop for iPad M1 Audit API."""
        try:
            while True:
                # 1. Static Telemetry Broadcast
                telemetry = {
                    "type": "telemetry",
                    "entropy": float(mx.sum(self.tensor_state).item()),
                    "status": "DENSE_MESH_ACTIVE"
                }
                await websocket.send(json.dumps(telemetry))

                # 2. UI Modulation Listener
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    user_msg = json.loads(raw_msg)
                    if self.ui_bridge and "query" in user_msg:
                        # Request schema from Sandbox
                        ui_schema = self.ui_bridge.generate_ui_payload(user_msg["query"])
                        await websocket.send(json.dumps(ui_schema))
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    pass

                await asyncio.sleep(0.05)
        except websockets.exceptions.ConnectionClosed:
            pass

    def start_server(self):
        server = websockets.serve(self._handle_connection, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()
