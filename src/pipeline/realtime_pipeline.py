import time
import cv2
import base64
import asyncio
import numpy as np
import threading

from src.camera.webcam_capture import WebcamCapture
from src.animation.portrait_3d_renderer import Portrait3DRenderer
from src.streaming.video_stream import broadcast, state

def _encode(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

class RealTimePipeline:
    def __init__(self, config: dict):
        self.config = config
        cam_cfg = config['camera']
        self.camera = WebcamCapture(cam_cfg['device_id'], cam_cfg['width'], cam_cfg['height'])
        self.renderer: Portrait3DRenderer | None = None
        self._portrait_id: int = 0

    def run(self):
        print("[Pipeline] Running...")
        fps = self.config['camera'].get('fps', 30)
        frame_time = 1.0 / fps

        while True:
            t0 = time.perf_counter()
            
            # Sync renderer with uploaded portrait
            if state.portrait_image is not None and id(state.portrait_image) != self._portrait_id:
                self._portrait_id = id(state.portrait_image)
                self.renderer = Portrait3DRenderer(state.portrait_image)

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Process / Render
            if state.simulation_active and self.renderer:
                out = self.renderer.render(frame)
            elif self.renderer:
                out = self.renderer.portrait
            else:
                out = np.zeros_like(frame)
                cv2.putText(out, "Upload Portrait", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Broadcast
            if state.loop and state.active_ws:
                payload = {
                    "live": _encode(cv2.resize(frame, (out.shape[1], out.shape[0]))),
                    "portrait": _encode(out),
                    "simulation_active": state.simulation_active
                }
                asyncio.run_coroutine_threadsafe(broadcast(payload), state.loop)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, frame_time - elapsed))
