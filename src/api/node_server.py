import json
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kernel.agi_kernel import JuniorAGI

app = FastAPI(title="JuniorAGI Sovereign Node API")
logger = logging.getLogger("JuniorAGI.API")

# Absolute instatiation to resolve Pyright undefined variable errors
sovereign_node = JuniorAGI()

@app.websocket("/ws/agi")
async def agi_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                if payload.get("type") == "inference_request":
                    prompt = payload.get("prompt", "")
                    result = await sovereign_node.submit_cognitive_task(prompt)
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
