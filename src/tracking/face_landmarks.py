import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceTracker:
    def __init__(self, model_path="models/face_landmarker.task", min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE # Use IMAGE mode for simplicity in this thread
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            return detection_result.face_landmarks[0]
        return None

    def draw_landmarks(self, frame, landmarks):
        if not landmarks:
            return frame
            
        h, w, _ = frame.shape
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        return frame
