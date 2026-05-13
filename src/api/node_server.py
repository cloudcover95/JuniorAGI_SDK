# src/api/node_server.py
import json
import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kernel.agi_kernel import JuniorAGI

logger = logging.getLogger("JuniorAGI.API")
logging.basicConfig(level=logging.INFO)

sovereign_node = JuniorAGI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[+] Initiating Sovereign Kernel Boot Sequence...")
    yield
    logger.info("[-] Executing Kernel Halt Sequence...")

app = FastAPI(title="JuniorAGI Sovereign Node API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/agi")
async def agi_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("[+] C2 WebSocket Connected.")
    
    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                req_type = payload.get("type")
                
                if req_type == "inference_request":
                    import mlx.core as mx
                    dims = sovereign_node.MODEL_PRESETS[sovereign_node.target_scale]["dims"]
                    local_dims = sovereign_node.mesh.shard_dimension(dims)
                    sim_tensor = mx.random.normal((1, 32, local_dims))
                    
                    result = sovereign_node.forward(sim_tensor)
                    result["shape"] = list(result.pop("y").shape) 
                    await websocket.send_text(json.dumps({"type": "inference_complete", "result": result}))
                    
        except WebSocketDisconnect:
            logger.info("[-] C2 Client disconnected.")

    receive_task = asyncio.create_task(receive_commands())
    try:
        while True:
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "telemetry": sovereign_node.economy.get_c2v_metrics()
            }))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        receive_task.cancel()
