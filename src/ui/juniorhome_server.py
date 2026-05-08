# ui/juniorhome_server.py
import logging
import asyncio
import json
from aiohttp import web
from typing import Any
from core.oauth_gateway import OAuth2Gateway

logger = logging.getLogger("JuniorOmega.HomeUI")

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JuniorOmega | AGI Command</title>
    <style>
        :root {
            --bg-base: #0d0e12; --bg-panel: #131722;
            --text-main: #d1d4dc; --text-dim: #787b86;
            --accent-green: #089981; --accent-red: #f23645; --accent-blue: #2962ff;
            --border: #2a2e39;
        }
        body { background-color: var(--bg-base); color: var(--text-main); font-family: monospace; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 24px; color: var(--text-main); }
        .status-indicator { display: flex; align-items: center; gap: 8px; color: var(--accent-green); }
        .status-dot { width: 10px; height: 10px; background-color: var(--accent-green); border-radius: 50%; box-shadow: 0 0 8px var(--accent-green); }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
        .panel { background-color: var(--bg-panel); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
        .panel h2 { margin: 0 0 12px 0; color: var(--accent-blue); text-transform: uppercase; font-size: 14px; border-bottom: 1px solid #1e222d; padding-bottom: 4px;}
        .data-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-size: 13px; }
        .data-key { color: var(--text-dim); }
        .log-container { background-color: #000; border: 1px solid var(--border); padding: 10px; height: 200px; overflow-y: auto; color: #a9b7c6; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>JuniorOmega :: AGI Command Matrix</h1>
        <div class="status-indicator"><div class="status-dot"></div><span id="conn-status">WS CONNECTED</span></div>
    </div>
    
    <div class="grid-container">
        <div class="panel">
            <h2>Procedural CAD (Morphogenesis)</h2>
            <div id="cad-data">Waiting for structural synthesis...</div>
        </div>
        
        <div class="panel">
            <h2>Logistics & Temporal CRON</h2>
            <div id="logistics-data">Waiting for schedule...</div>
        </div>

        <div class="panel">
            <h2>Omni-Matrix & Economy</h2>
            <div id="economy-data">Waiting...</div>
        </div>
        
        <div class="panel" style="grid-column: 1 / -1;">
            <h2>JuniorOmega Core Telemetry</h2>
            <div class="log-container" id="mesh-logs"></div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if(data.JuniorOmega_Structural_Synthesis) {
                const cad = data.JuniorOmega_Structural_Synthesis;
                document.getElementById('cad-data').innerHTML = `
                    <div class="data-row"><span class="data-key">Meshes Synthesized:</span><span class="data-val">${cad.meshes_synthesized}</span></div>
                    <div class="data-row"><span class="data-key">Total Vertices:</span><span class="data-val">${cad.total_vertices_generated}</span></div>
                    <div class="data-row"><span class="data-key">Status:</span><span class="data-val" style="color:var(--accent-green)">ONLINE</span></div>
                `;
            }

            if(data.TemporalCRON && data.Sovereign_Logistics_Matrix) {
                document.getElementById('logistics-data').innerHTML = `
                    <div class="data-row"><span class="data-key">Pending CRON Tasks:</span><span class="data-val">${data.TemporalCRON.pending_tasks}</span></div>
                    <div class="data-row"><span class="data-key">ETL Jobs Processed:</span><span class="data-val">${data.Sovereign_Logistics_Matrix.etl_jobs}</span></div>
                    <div class="data-row"><span class="data-key">SMTP Pushed:</span><span class="data-val">${data.Sovereign_Logistics_Matrix.emails_sent}</span></div>
                `;
            }

            if(data.InternalEconomy) {
                document.getElementById('economy-data').innerHTML = `
                    <div class="data-row"><span class="data-key">Compute Budget:</span> <span class="data-val">${data.InternalEconomy.current_budget} T/s</span></div>
                    <div class="data-row"><span class="data-key">Cloud APIs:</span> <span class="data-val" style="color:var(--accent-green)">ROUTING ENABLED</span></div>
                `;
            }
        };
    </script>
</body>
</html>
"""

class JuniorHomeServer:
    def __init__(self, agi_node: Any, oauth_gateway: OAuth2Gateway, port: int = 8080):
        self.node = agi_node
        self.oauth = oauth_gateway
        self.port = port
        self.app = web.Application()
        
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        self.runner = web.AppRunner(self.app)
        self._ws_clients = set()
        self._running = False

    async def handle_index(self, request):
        return web.Response(text=HTML_DASHBOARD, content_type='text/html')

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        try:
            async for msg in ws: pass
        finally:
            self._ws_clients.remove(ws)
        return ws

    async def start(self):
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        await site.start()
        self._running = True
        logger.info(f"[+] JuniorOmega UI hosted on 0.0.0.0:{self.port}.")
        asyncio.create_task(self._broadcast_telemetry())

    async def _broadcast_telemetry(self):
        while self._running:
            if self._ws_clients:
                telemetry = self.node.get_system_telemetry()
                payload = json.dumps(telemetry)
                for ws in list(self._ws_clients):
                    try: await ws.send_str(payload)
                    except Exception: self._ws_clients.remove(ws)
            await asyncio.sleep(1.0)

    async def stop(self):
        self._running = False
        await self.runner.cleanup()
