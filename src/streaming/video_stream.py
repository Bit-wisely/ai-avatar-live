from fastapi import FastAPI, WebSocket
import asyncio
import cv2
import base64

app = FastAPI()

class VideoStreamer:
    """
    Handles streaming the rendered frames over WebSockets.
    """
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def stream_frame(self, frame):
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        for connection in self.active_connections:
            try:
                await connection.send_text(jpg_as_text)
            except:
                self.disconnect(connection)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    streamer = VideoStreamer()
    await streamer.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1) # Keep connection alive
    except:
        streamer.disconnect(websocket)
