"""
Driver Awareness Relay Server
==============================
Two WebSocket endpoints:

  wss://<host>/ws/sensor   <- face_analysis.py connects here and pushes readings
  wss://<host>/ws/app      <- your application connects here and receives readings

  GET /status              <- latest reading as JSON (REST fallback)
  GET /health              <- health check for Render
"""

import json
import time
import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Driver Awareness Relay")

# Allow all origins so any client app can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared latest state ───────────────────────────────────────────────────────
latest: dict = {
    "score":       100.0,
    "alert_level": 0,
    "message":     "",
    "dominant":    "",
    "s_perclos":   100.0,
    "s_turn":      100.0,
    "s_tilt":      100.0,
    "perclos_pct": 0.0,
    "yaw":         0.0,
    "roll":        0.0,
    "timestamp":   0.0,
    "sensor_connected": False,
}

# Connected app clients
app_clients: list[WebSocket] = []
sensor_connected = False


async def broadcast(payload: dict):
    """Send payload to all connected app clients, remove dead ones."""
    dead = []
    for client in app_clients:
        try:
            await client.send_text(json.dumps(payload))
        except Exception:
            dead.append(client)
    for d in dead:
        app_clients.remove(d)


# ── /ws/sensor  — face_analysis.py pushes here ───────────────────────────────
@app.websocket("/ws/sensor")
async def sensor_endpoint(ws: WebSocket):
    global sensor_connected
    await ws.accept()
    sensor_connected = True
    latest["sensor_connected"] = True
    print("[sensor] face_analysis.py connected", flush=True)

    # Notify all app clients that sensor came online
    await broadcast({**latest, "event": "sensor_connected"})

    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)
            latest.update(payload)
            latest["timestamp"] = time.time()
            latest["sensor_connected"] = True
            await broadcast(latest)
    except WebSocketDisconnect:
        sensor_connected = False
        latest["sensor_connected"] = False
        print("[sensor] face_analysis.py disconnected", flush=True)
        await broadcast({**latest, "event": "sensor_disconnected"})


# ── /ws/app  — your application connects here ─────────────────────────────────
@app.websocket("/ws/app")
async def app_endpoint(ws: WebSocket):
    await ws.accept()
    app_clients.append(ws)
    print(f"[app] client connected  (total: {len(app_clients)})", flush=True)

    # Send current state immediately on connect
    await ws.send_text(json.dumps(latest))

    try:
        while True:
            # Keep connection alive; ignore messages from app side
            await asyncio.wait_for(ws.receive_text(), timeout=30)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if ws in app_clients:
            app_clients.remove(ws)
        print(f"[app] client disconnected  (total: {len(app_clients)})", flush=True)


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    """Latest reading as JSON — for polling or debugging."""
    return latest


@app.get("/health")
def health():
    """Render health check."""
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
