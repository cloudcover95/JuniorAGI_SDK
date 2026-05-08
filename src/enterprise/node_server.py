# enterprise/node_server.py
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws/agi")
async def agi_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    async def receive_commands():
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                if payload["type"] == "inference_request":
                    # C2 Preemption: Priority 0 (Absolute)
                    asyncio.create_task(run_inference_stream(payload["prompt"]))
        except WebSocketDisconnect:
            pass

    async def run_inference_stream(prompt: str):
        async with sovereign_node.inference_lock:
            try:
                # Use generate_stream_async to allow concurrent swarm offloading
                async for chunk in sovereign_node.generate_stream_async(prompt, priority=0):
                    out_payload = {
                        "type": "token",
                        "chunk": chunk,
                        "telemetry": sovereign_node.get_system_telemetry()
                    }
                    await websocket.send_text(json.dumps(out_payload))
                await websocket.send_text(json.dumps({"type": "inference_complete"}))
            except Exception as e:
                await websocket.send_text(json.dumps({"type": "error", "msg": str(e)}))

    receive_task = asyncio.create_task(receive_commands())
    try:
        while True:
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "telemetry": sovereign_node.get_system_telemetry()
            }))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        receive_task.cancel()
