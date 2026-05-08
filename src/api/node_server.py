# src/api/node_server.py
import json
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kernel.agi_kernel import JuniorAGI

logger = logging.getLogger("JuniorAGI.API")
logging.basicConfig(level=logging.INFO)

sovereign_node = JuniorAGI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ignition
    logger.info("[+] Initiating Sovereign Kernel Boot Sequence...")
    await sovereign_node.boot_sequence()
    yield
    # Halt
    logger.info("[-] Executing Kernel Halt Sequence...")
    await sovereign_node.halt_sequence()

app = FastAPI(title="JuniorAGI Sovereign Node API", lifespan=lifespan)

@app.websocket("/ws/agi")
async def agi_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                if payload.get("type") == "inference_request":
                    result = await sovereign_node.submit_cognitive_task(payload.get("prompt", ""))
                    await websocket.send_text(json.dumps({"type": "inference_complete", "result": result}))
        except WebSocketDisconnect:
            pass

    receive_task = asyncio.create_task(receive_commands())
    try:
        while True:
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "telemetry": sovereign_node.get_telemetry()
            }))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        receive_task.cancel()
