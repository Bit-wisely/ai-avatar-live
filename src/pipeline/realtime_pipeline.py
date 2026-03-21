"""
RealTimePipeline – revised
==========================
* Accepts dynamically-supplied portrait (from HTTP upload)
* Sends both the LIVE webcam frame AND the animated 3-D portrait frame
  to WebSocket clients as a single JSON message
* Supports start / stop driven by the simulation_active flag in AppState
"""

import threading
import time
import cv2
import base64
import asyncio
import numpy as np

from src.camera.webcam_capture import WebcamCapture
from src.tracking.face_landmarks import FaceTracker
from src.animation.portrait_3d_renderer import Portrait3DRenderer
from src.streaming.video_stream import streamer, state


def _encode_frame(frame: np.ndarray, quality: int = 82) -> str:
    """BGR numpy → base64 JPEG string."""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


class RealTimePipeline:
    """
    Captures webcam, tracks face, warps portrait, and streams both feeds.
    """

    def __init__(self, config: dict):
        self.config = config
        self.running = False

        cam_cfg = config['camera']
        self.camera = WebcamCapture(
            cam_cfg['device_id'],
            cam_cfg['width'],
            cam_cfg['height'],
        )
        self.tracker = FaceTracker()

        # Portrait renderer – created lazily when a portrait is loaded
        self._renderer: Portrait3DRenderer | None = None
        self._renderer_lock = threading.Lock()

        # Background watcher thread checks for new portraits
        self._portrait_hash: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self):
        """Main blocking loop – call from a dedicated thread."""
        self.running = True
        fps_target = self.config['camera'].get('fps', 30)
        frame_time = 1.0 / fps_target

        print("[Pipeline] Starting capture loop …")

        while self.running:
            t0 = time.perf_counter()

            # 1. Check if a new portrait has been uploaded
            self._refresh_renderer()

            # 2. Grab live frame
            live_frame = self.camera.get_frame()
            if live_frame is None:
                time.sleep(0.05)
                continue

            # 3. Only run the expensive render if simulation is active
            if state.simulation_active and self._renderer is not None:
                portrait_out = self._renderer.render(live_frame)
            elif self._renderer is not None:
                # Show static portrait when paused
                portrait_out = self._renderer.portrait.copy()
            else:
                # No portrait yet – show placeholder
                portrait_out = self._make_placeholder(live_frame.shape)

            # 4. Resize live frame to match portrait size for side-by-side display
            ph, pw = portrait_out.shape[:2]
            live_resized = cv2.resize(live_frame, (pw, ph))

            # 5. Broadcast both frames as a single JSON message
            payload = {
                "live": _encode_frame(live_resized),
                "portrait": _encode_frame(portrait_out),
                "simulation_active": state.simulation_active,
            }

            if state.loop:
                import json
                asyncio.run_coroutine_threadsafe(
                    streamer.broadcast_json(payload),
                    state.loop
                )

            # 6. Rate limit
            elapsed = time.perf_counter() - t0
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)

        print("[Pipeline] Capture loop stopped.")

    def stop(self):
        self.running = False
        self.camera.release()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _refresh_renderer(self):
        """Re-create renderer if the shared portrait has changed."""
        portrait = state.portrait_image
        if portrait is None:
            return
        h = id(portrait)
        if h != self._portrait_hash:
            self._portrait_hash = h
            new_renderer = Portrait3DRenderer(portrait)
            with self._renderer_lock:
                self._renderer = new_renderer
            print("[Pipeline] Renderer refreshed with new portrait.")

    @staticmethod
    def _make_placeholder(live_shape: tuple) -> np.ndarray:
        h, w = live_shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 25)   # dark background
        text = "Upload a portrait"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7 if w > 400 else 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        tx, ty = (w - tw) // 2, (h + th) // 2
        cv2.putText(canvas, text, (tx, ty), font, scale, (120, 120, 160), thickness, cv2.LINE_AA)
        return canvas
