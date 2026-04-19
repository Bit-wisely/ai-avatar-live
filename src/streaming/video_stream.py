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

class AppState:
    def __init__(self):
        self.portrait_image: np.ndarray | None = None
        self.portrait_b64: str = ""
        self.simulation_active: bool = False
        self.active_ws: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

state = AppState()

async def broadcast(payload: dict):
    """Send JSON payload to all connected clients."""
    if not state.active_ws:
        return
    msg = json.dumps(payload)
    for ws in list(state.active_ws):
        try:
            await ws.send_text(msg)
        except Exception:
            if ws in state.active_ws:
                state.active_ws.remove(ws)

@app.get("/")
async def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/upload_portrait")
async def upload_portrait(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    state.portrait_image = cv2.resize(img, (512, 512))
    _, buf = cv2.imencode('.jpg', state.portrait_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    state.portrait_b64 = base64.b64encode(buf).decode('utf-8')

    await broadcast({"event": "portrait_loaded", "portrait": state.portrait_b64})
    return {"status": "ok"}

@app.get("/status")
async def get_status():
    return {
        "simulation_active": state.simulation_active,
        "portrait_loaded": state.portrait_image is not None,
        "connected_clients": len(state.active_ws),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_ws.append(websocket)
    if state.loop is None:
        state.loop = asyncio.get_running_loop()

    # Initial state
    init_msg = {"event": "init", "simulation_active": state.simulation_active}
    if state.portrait_b64:
        init_msg["portrait"] = state.portrait_b64
    await websocket.send_text(json.dumps(init_msg))

    try:
        while True:
            data = await websocket.receive_text()
    except Exception:
        streamer.disconnect(websocket)
