import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

app = FastAPI(title="Driver Awareness Relay (WebSocket)")

# Allow all clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ─────────────────────────────────────────────
latest = {
    "score": 99.0,
    "alert_level": 0,
    "message": "",
    "dominant": "",
    "s_perclos": 100.0,
    "s_turn": 100.0,
    "s_tilt": 100.0,
    "perclos_pct": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "timestamp": 0.0,
    "sensor_connected": False,
}

# ── WebSocket endpoint ──────────────────────────────────────
@app.websocket("/ws/sensor")
async def websocket_sensor(websocket: WebSocket):
    """WebSocket endpoint to receive face analysis data"""
    await websocket.accept()
    print(f"[WS] Client connected from {websocket.client}", flush=True)
    latest["sensor_connected"] = True
    
    try:
        while True:
            # Receive data from face_analysis
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # Update latest state with received data
            latest.update(payload)
            latest["sensor_connected"] = True
            
            print(f"[WS] Updated latest: Score={payload.get('score')}, Alert={payload.get('alert_level')}", flush=True)
    
    except WebSocketDisconnect:
        print("[WS] Client disconnected", flush=True)
        latest["sensor_connected"] = False
    except json.JSONDecodeError as e:
        print(f"[WS] JSON error: {e}", flush=True)
        latest["sensor_connected"] = False
    except Exception as e:
        print(f"[WS] Error: {type(e).__name__}: {e}", flush=True)
        latest["sensor_connected"] = False

# ── SENSOR pushes data here (HTTP fallback) ────────────────
@app.post("/sensor")
async def sensor_update(payload: dict):
    latest.update(payload)
    latest["timestamp"] = time.time()
    latest["sensor_connected"] = True
    return {"status": "received"}


# ── APP pulls data here ─────────────────────────────────────
@app.get("/status")
def get_status():
    return latest


# ── Health check ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "connected": latest["sensor_connected"],
        "latest_score": latest.get("score")
    }


# ── Run server ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Driver Awareness Server starting...")
    print("📡 WebSocket: ws://127.0.0.1:8000/ws/sensor")
    print("📊 Status: http://127.0.0.1:8000/status")
    print("❤️  Health: http://127.0.0.1:8000/health")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)