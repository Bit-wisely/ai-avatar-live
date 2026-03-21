import os
import asyncio
import cv2
import base64
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Shared application state ─────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.portrait_image: np.ndarray | None = None   # uploaded portrait (BGR)
        self.portrait_b64: str = ""                      # base64 for initial send
        self.simulation_active: bool = False
        self.active_ws: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

state = AppState()


class VideoStreamer:
    """Streams rendered frames to all connected WebSocket clients."""

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        state.active_ws.append(websocket)
        if state.loop is None:
            state.loop = asyncio.get_running_loop()

    def disconnect(self, websocket: WebSocket):
        if websocket in state.active_ws:
            state.active_ws.remove(websocket)

    async def broadcast_json(self, payload: dict):
        msg = json.dumps(payload)
        for ws in list(state.active_ws):
            try:
                await ws.send_text(msg)
            except Exception:
                self.disconnect(ws)

    async def broadcast_frame(self, frame: np.ndarray, key: str = "avatar_frame"):
        """Encode frame as JPEG and broadcast as b64 JSON."""
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode('utf-8')
        await self.broadcast_json({key: b64})


streamer = VideoStreamer()

# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/upload_portrait")
async def upload_portrait(file: UploadFile = File(...)):
    """Receive a portrait image upload and store it in memory."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Resize to a fixed square for 3D rendering (512 × 512)
    img_resized = cv2.resize(img, (512, 512))
    state.portrait_image = img_resized

    # Also create b64 preview to send to clients
    _, buf = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    state.portrait_b64 = base64.b64encode(buf).decode('utf-8')

    # Notify all connected clients that a new portrait is ready
    await streamer.broadcast_json({"event": "portrait_loaded", "portrait": state.portrait_b64})
    return JSONResponse({"status": "ok", "message": "Portrait loaded successfully"})


@app.get("/status")
async def get_status():
    return {
        "simulation_active": state.simulation_active,
        "portrait_loaded": state.portrait_image is not None,
        "connected_clients": len(state.active_ws),
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await streamer.connect(websocket)

    # Send initial state
    init_payload: dict = {"event": "init", "simulation_active": state.simulation_active}
    if state.portrait_b64:
        init_payload["portrait"] = state.portrait_b64
    await websocket.send_text(json.dumps(init_payload))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("cmd")
                if cmd == "start":
                    state.simulation_active = True
                    await streamer.broadcast_json({"event": "started"})
                elif cmd == "stop":
                    state.simulation_active = False
                    await streamer.broadcast_json({"event": "stopped"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        streamer.disconnect(websocket)
    except Exception:
        streamer.disconnect(websocket)
