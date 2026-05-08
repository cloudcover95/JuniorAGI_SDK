# ui/agi_dashboard.py
# Rebranded JuniorAGI Command UI
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>JuniorAGI | Sovereign SDK</title>
    <style>
        :root { --bg: #050505; --panel: #0a0a0a; --accent: #2962ff; --text: #e0e0e0; --dim: #666; --green: #00c853; }
        body { background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; border-bottom: 1px solid #111; padding-bottom: 15px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .panel { background: var(--panel); border: 1px solid #111; padding: 15px; border-radius: 4px; }
        .panel h2 { color: var(--accent); font-size: 12px; text-transform: uppercase; margin-top: 0; border-bottom: 1px solid #111; padding-bottom: 5px; }
        .row { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 5px; }
        .val { color: var(--green); }
    </style>
</head>
<body>
    <div class="header">
        <div style="font-size: 18px; font-weight: bold;">JuniorAGI SDK <span style="color:var(--dim); font-weight:normal;">v0.47.0</span></div>
        <div id="git-status" style="font-size: 11px; color: var(--dim);">GIT: SYNCED</div>
    </div>
    <div class="grid">
        <div class="panel">
            <h2>Sovereign Git Matrix</h2>
            <div id="git-data"></div>
        </div>
        <div class="panel">
            <h2>Chrono-Logistics</h2>
            <div id="logistics-data"></div>
        </div>
        <div class="panel">
            <h2>Cognitive Economy</h2>
            <div id="economy-data"></div>
        </div>
    </div>
</body>
</html>
"""
