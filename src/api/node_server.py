import json, asyncio, logging, sys, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from kernel.agi_kernel import JuniorAGI

logger = logging.getLogger("JuniorAGI.API")
agi = JuniorAGI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Kernel Booted.")
    yield
    logger.info("Kernel Halted.")

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/agi")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async def rx():
        try:
            while True:
                data = json.loads(await ws.receive_text())
                if data.get("type") == "inference":
                    import mlx.core as mx
                    x = mx.random.normal((1, 32, agi.PRESETS[agi.scale][0]))
                    res = agi.forward(x, cmd=data.get("tool"))
                    res["shape"] = list(res.pop("y").shape)
                    await ws.send_text(json.dumps({"type": "result", "data": res}))
        except WebSocketDisconnect: pass
    t = asyncio.create_task(rx())
    try:
        while True:
            await ws.send_text(json.dumps({"type": "telemetry", "data": agi.eco.get_c2v_metrics()}))
            await asyncio.sleep(2)
    except WebSocketDisconnect: t.cancel()
