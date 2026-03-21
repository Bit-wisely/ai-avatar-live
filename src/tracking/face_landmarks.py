"""
FaceTracker – updated for MediaPipe Tasks API (mediapipe >= 0.10)
The main rendering pipeline now uses Portrait3DRenderer directly,
so this module is retained for optional debug overlays.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "face_landmarker.task"
)
_MODEL_PATH = os.path.normpath(_MODEL_PATH)


class FaceTracker:
    def __init__(self, model_path: str = _MODEL_PATH,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)

    def process(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def draw_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        if not landmarks:
            return frame
        h, w = frame.shape[:2]
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame
